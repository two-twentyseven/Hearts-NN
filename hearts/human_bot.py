"""Deterministic heuristic Hearts bot.

Plays from ``EncodedObservation`` (the same public view as the net) plus a
legal mask. Heuristics follow common hold-round advice: count cards, flush or
dump the Queen of Spades, duck point tricks, build voids, and stop an obvious
moon. Passing uses a separate decision tree (spade security, then danger suits).
The bot never tries to shoot the moon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from hearts.cards import Card, Rank, Suit
from hearts.config import GameConfig
from hearts.encode import NONE_INDEX, POINTS_PER_ROUND, EncodedObservation, encode_state
from hearts.env import HoldRoundEnv, card_from_id
from hearts.model import legal_mask_from_cards
from hearts.rules import NUM_PLAYERS, legal_plays
from hearts.state import DECK_SIZE, Phase
from hearts.train.corpus import DecisionRecord, records_to_batch

# Diamonds, clubs, hearts, spades — used to break ties among equal ranks.
_TIE_SUIT = {
    Suit.DIAMONDS: 0,
    Suit.CLUBS: 1,
    Suit.HEARTS: 2,
    Suit.SPADES: 3,
}
_SUITS = (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES)
_LOW_MAX = Rank.SIX
_LATE_TRICK = 9  # 0-based trick index 9+ (tricks 10–13)
_QS_COVER = 2
_PASS_COUNT = 3
_BUFFER_MAX = Rank.SEVEN
_DANGER_MIN = Rank.EIGHT
_MIN_SPADE_COVER = 3
_DANGER_SUIT_PRIORITY = {
    Suit.HEARTS: 2,
    Suit.DIAMONDS: 1,
    Suit.CLUBS: 0,
}

__all__ = [
    "WorldView",
    "TrickPlay",
    "decode_observation",
    "select_card",
    "select_pass",
    "policy_from_choice",
    "play_heuristic_hold_round",
]


@dataclass(frozen=True)
class TrickPlay:
    """One card already on the current trick, from the acting seat's view."""

    seat: int
    card: Card
    position: int


@dataclass(frozen=True)
class WorldView:
    """Public facts reconstructed from ``EncodedObservation``."""

    hand: tuple[Card, ...]
    suit_led: Suit | None
    my_role: int
    is_leading: bool
    is_last: bool
    hearts_broken: bool
    current_trick: int
    is_first_trick: bool
    points_taken: tuple[int, int, int, int]
    qs_in_hand: bool
    qs_played: bool
    qs_played_by: int | None
    voids: tuple[frozenset[Suit], ...]
    current_plays: tuple[TrickPlay, ...]
    winning_card: Card | None
    winning_seat: int | None
    trick_has_points: bool
    trick_has_qs: bool
    played_ids: frozenset[int]
    remaining_by_suit: tuple[int, int, int, int]
    opponent_has_suit: tuple[bool, bool, bool, bool]
    suit_lead_counts: tuple[int, int, int, int]
    shown_voids: tuple[frozenset[Suit], ...]


def decode_observation(obs: EncodedObservation) -> WorldView:
    """Turn card/global feature arrays into the facts the tree branches on."""
    hand = tuple(
        card_from_id(card_id)
        for card_id in range(DECK_SIZE)
        if obs.in_hand[card_id] == 1.0
    )
    suit_led = None if obs.suit_led == NONE_INDEX else _SUITS[int(obs.suit_led)]
    hearts_broken = bool(obs.hearts_broken)
    current_trick = int(obs.current_trick)
    my_role = int(obs.my_role)
    points = tuple(int(round(float(p) * POINTS_PER_ROUND)) for p in obs.points_taken)

    played_ids = frozenset(
        card_id for card_id in range(DECK_SIZE) if obs.played[card_id] == 1.0
    )
    qs = Card(Rank.QUEEN, Suit.SPADES)
    qs_in_hand = any(c.is_queen_of_spades for c in hand)
    qs_played = qs.card_id in played_ids
    qs_played_by = int(obs.played_by[qs.card_id]) if qs_played else None
    if qs_played_by is not None and qs_played_by == NONE_INDEX:
        qs_played_by = None

    plays = []
    for card_id in range(DECK_SIZE):
        if obs.played[card_id] != 1.0:
            continue
        if int(obs.played_on_trick[card_id]) != current_trick:
            continue
        plays.append(
            TrickPlay(
                seat=int(obs.played_by[card_id]),
                card=card_from_id(card_id),
                position=int(obs.play_position[card_id]),
            )
        )
    plays.sort(key=lambda play: play.position)

    winning_card: Card | None = None
    winning_seat: int | None = None
    if suit_led is not None:
        for play in plays:
            if play.card.suit is not suit_led:
                continue
            if winning_card is None or play.card.rank.value > winning_card.rank.value:
                winning_card = play.card
                winning_seat = play.seat

    trick_has_qs = any(play.card.is_queen_of_spades for play in plays)
    trick_has_points = trick_has_qs or any(play.card.is_heart for play in plays)

    voids: list[set[Suit]] = [set() for _ in range(NUM_PLAYERS)]
    shown_voids: list[set[Suit]] = [set() for _ in range(NUM_PLAYERS)]
    held_suits = {card.suit for card in hand}
    for suit in _SUITS:
        if suit not in held_suits:
            voids[0].add(suit)
    for card_id in range(DECK_SIZE):
        if obs.broke_suit[card_id] != 1.0:
            continue
        trick_index = int(obs.played_on_trick[card_id])
        led = int(obs.trick_led_suits[trick_index])
        if led == NONE_INDEX:
            continue
        seat = int(obs.played_by[card_id])
        shown = _SUITS[led]
        voids[seat].add(shown)
        shown_voids[seat].add(shown)

    lead_counts = [0, 0, 0, 0]
    for led in obs.trick_led_suits:
        idx = int(led)
        if idx != NONE_INDEX:
            lead_counts[idx] += 1

    remaining: list[int] = []
    opponent_has: list[bool] = []
    for suit in _SUITS:
        start = (suit.value - 1) * 13
        played_count = sum(
            1 for rank_i in range(13) if (start + rank_i) in played_ids
        )
        ours = sum(1 for card in hand if card.suit is suit)
        left = 13 - played_count
        remaining.append(left)
        opp_left = left - ours
        opponent_has.append(opp_left > 0)
        if opp_left <= 0:
            for seat in range(1, NUM_PLAYERS):
                voids[seat].add(suit)

    return WorldView(
        hand=hand,
        suit_led=suit_led,
        my_role=my_role,
        is_leading=suit_led is None,
        is_last=my_role == NUM_PLAYERS - 1,
        hearts_broken=hearts_broken,
        current_trick=current_trick,
        is_first_trick=current_trick == 0,
        points_taken=points,
        qs_in_hand=qs_in_hand,
        qs_played=qs_played,
        qs_played_by=qs_played_by,
        voids=tuple(frozenset(v) for v in voids),
        current_plays=tuple(plays),
        winning_card=winning_card,
        winning_seat=winning_seat,
        trick_has_points=trick_has_points,
        trick_has_qs=trick_has_qs,
        played_ids=played_ids,
        remaining_by_suit=tuple(remaining),
        opponent_has_suit=tuple(opponent_has),
        suit_lead_counts=tuple(lead_counts),
        shown_voids=tuple(frozenset(v) for v in shown_voids),
    )


def select_card(obs: EncodedObservation, legal_mask: Tensor | np.ndarray) -> Card:
    """Pick one legal card from the observation via the heuristic tree."""
    view = decode_observation(obs)
    legal = _legal_cards(view, legal_mask)
    if not legal:
        raise ValueError("No legal cards")
    if len(legal) == 1:
        return legal[0]

    stopped = _moon_stop_play(view, legal)
    if stopped is not None:
        return stopped
    # First two club/diamond leads, nobody shown void: shed the highest.
    # Hearts stay duck-first (points). Spades keep Queen/Ace/King rules.
    if (
        not view.is_leading
        and _can_follow(view)
        and view.suit_led in (Suit.CLUBS, Suit.DIAMONDS)
        and _early_safe_suit(view, view.suit_led)
    ):
        return _highest_of(legal)
    if view.is_leading:
        return _lead_card(view, legal)
    if _can_follow(view):
        return _follow_card(view, legal)
    return _sluff_card(view, legal)


def policy_from_choice(card: Card) -> Tensor:
    """One-hot 52-way target for corpus imitation."""
    pi = torch.zeros(DECK_SIZE, dtype=torch.float32)
    pi[card.card_id] = 1.0
    return pi


def play_heuristic_hold_round(seed: int | None = None) -> dict[str, Tensor]:
    """Play one hold round with four heuristic seats; return a corpus batch."""
    env = HoldRoundEnv()
    state = env.reset(seed=seed)
    records: list[DecisionRecord] = []
    while state.phase is Phase.PLAYING:
        acting = state.current_player
        legal = state.legal_moves_for(acting)
        obs = encode_state(state, acting)
        mask = legal_mask_from_cards(legal)
        card = select_card(obs, mask)
        records.append(
            DecisionRecord(
                obs=obs,
                legal_mask=mask.clone(),
                pi_target=policy_from_choice(card),
                acting_player=acting,
            )
        )
        state = env.step(card)
    return records_to_batch(records, env.returns_by_player())


def select_pass(hand: list[Card] | tuple[Card, ...]) -> list[Card]:
    """Pick three cards to pass. Never tries to shoot the moon.

    Decision tree:
    1. Secure spades first (AK unguarded, then unprotected QS). Never pass
       other spades unless the hand has nothing else left.
    2. Dump 8+ cards from danger suits (0–2 buffers of rank 7 or lower),
       ranking suits by average rank and taking the highest danger cards.
    3. Fill from the shortest remaining non-spade suits (voids), highest first.
    """
    cards = list(hand)
    if len(cards) < _PASS_COUNT:
        raise ValueError("Hand must contain at least three cards")
    chosen: list[Card] = []

    def take(card: Card) -> None:
        if card not in chosen and len(chosen) < _PASS_COUNT:
            chosen.append(card)

    for card in _spade_security_passes(cards):
        take(card)
    remaining = [card for card in cards if card not in chosen]
    for card in _danger_suit_passes(remaining):
        take(card)
    remaining = [card for card in cards if card not in chosen]
    for card in _fill_pass(remaining):
        take(card)
    if len(chosen) < _PASS_COUNT:
        raise ValueError("Could not select three cards to pass")
    return chosen[:_PASS_COUNT]


# --- decode helpers / legal set -------------------------------------------------


def _legal_cards(view: WorldView, legal_mask: Tensor | np.ndarray) -> list[Card]:
    mask = _mask_bool(legal_mask)
    mask_cards = [card_from_id(i) for i in range(DECK_SIZE) if mask[i]]
    mask_ids = {card.card_id for card in mask_cards}
    reconstructed = legal_plays(
        list(view.hand),
        is_leading=view.is_leading,
        led_suit=view.suit_led,
        hearts_broken=view.hearts_broken,
        is_first_trick=view.is_first_trick,
        config=GameConfig(),
    )
    legal = [card for card in reconstructed if card.card_id in mask_ids]
    return legal if legal else mask_cards


def _mask_bool(legal_mask: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(legal_mask, Tensor):
        return legal_mask.detach().cpu().numpy().astype(bool)
    return np.asarray(legal_mask).astype(bool)


def _can_follow(view: WorldView) -> bool:
    return view.suit_led is not None and any(
        card.suit is view.suit_led for card in view.hand
    )


# --- passing --------------------------------------------------------------------


def _spades_in(hand: list[Card]) -> list[Card]:
    return [card for card in hand if card.suit is Suit.SPADES]


def _spade_security_passes(hand: list[Card]) -> list[Card]:
    """Unguarded AS/KS, then QS if remaining cover is under three.

    Spade danger is Q–A; Jack and lower are buffers. For the Queen, every
    other spade (including AK) counts as cover. Low spades are never passed
    at this step.
    """
    spades = _spades_in(hand)
    ace = next((card for card in spades if card.rank is Rank.ACE), None)
    king = next((card for card in spades if card.rank is Rank.KING), None)
    queen = next((card for card in spades if card.is_queen_of_spades), None)
    honors = [card for card in (ace, king) if card is not None]
    other_than_ak = [
        card for card in spades if card.rank not in (Rank.ACE, Rank.KING)
    ]

    out: list[Card] = []
    if honors and len(other_than_ak) < _MIN_SPADE_COVER:
        out.extend(honors)

    remaining = [card for card in spades if card not in out]
    cover = [card for card in remaining if not card.is_queen_of_spades]
    if queen is not None and queen not in out and len(cover) < _MIN_SPADE_COVER:
        out.append(queen)
    return out


def _is_danger_suit(cards: list[Card]) -> bool:
    if not cards:
        return False
    buffers = [card for card in cards if card.rank.value <= _BUFFER_MAX.value]
    danger = [card for card in cards if card.rank.value >= _DANGER_MIN.value]
    return bool(danger) and len(buffers) <= 2


def _suit_avg_cost(cards: list[Card]) -> float:
    return sum(card.rank.value for card in cards) / len(cards)


def _danger_suit_passes(hand: list[Card]) -> list[Card]:
    """Highest 8+ cards from danger suits, worst average cost first."""
    ranked: list[tuple[float, int, list[Card]]] = []
    for suit in (Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS):
        cards = [card for card in hand if card.suit is suit]
        if not _is_danger_suit(cards):
            continue
        danger = [card for card in cards if card.rank.value >= _DANGER_MIN.value]
        danger.sort(key=lambda card: card.rank.value, reverse=True)
        ranked.append((_suit_avg_cost(cards), _DANGER_SUIT_PRIORITY[suit], danger))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    out: list[Card] = []
    for _, _, danger in ranked:
        out.extend(danger)
    return out


def _fill_pass(hand: list[Card]) -> list[Card]:
    """Void short non-spade suits, then highest leftovers; spades last."""
    remaining = list(hand)
    out: list[Card] = []
    while remaining:
        non_spade = [card for card in remaining if card.suit is not Suit.SPADES]
        pool = non_spade or remaining
        if non_spade:
            lengths: dict[Suit, list[Card]] = {}
            for card in non_spade:
                lengths.setdefault(card.suit, []).append(card)
            suit = min(
                lengths,
                key=lambda item: (
                    len(lengths[item]),
                    -max(card.rank.value for card in lengths[item]),
                ),
            )
            pick = _highest_of(lengths[suit])
        else:
            # Forced to pass a spade: keep QS if anything else remains.
            non_qs = [card for card in pool if not card.is_queen_of_spades]
            pick = _lowest_of(non_qs or pool)
        out.append(pick)
        remaining = [card for card in remaining if card is not pick]
    return out


# --- moon stop (never shoot; only block an opponent) ----------------------------


def _opponent_has_all_points(view: WorldView) -> int | None:
    """Relative seat of the unique non-zero point holder, if it is not us."""
    holders = [seat for seat, pts in enumerate(view.points_taken) if pts > 0]
    if len(holders) != 1:
        return None
    shooter = holders[0]
    if shooter == 0:
        return None
    return shooter


def _moon_stop_play(view: WorldView, legal: list[Card]) -> Card | None:
    shooter = _opponent_has_all_points(view)
    if shooter is None:
        return None
    safe = [card for card in legal if not card.is_queen_of_spades]
    if not safe:
        return None

    if view.is_leading:
        hearts = [card for card in safe if card.is_heart]
        if hearts:
            return _highest_of(hearts)
        return None

    if _can_follow(view):
        over = _over(safe, view.winning_card)
        if view.winning_seat == shooter and over:
            return _highest_of(over)
        if view.trick_has_points:
            under = _under(safe, view.winning_card)
            if under:
                return _highest_of(under)
        return None

    hearts = [card for card in safe if card.is_heart]
    if hearts and view.winning_seat != shooter:
        return _lowest_of(hearts)
        return None


# --- lead -----------------------------------------------------------------------


def _lead_card(view: WorldView, legal: list[Card]) -> Card:
    if all(card.is_heart for card in legal):
        return _lowest_of(legal)
    if view.is_first_trick:
        return _lowest_of(legal)

    pool = _without_spade_danger(view, legal)

    if view.qs_in_hand:
        void_lead = _lead_to_void(view, pool)
        if void_lead is not None:
            return void_lead

    if not view.qs_in_hand and not view.qs_played:
        low_spades = [
            card
            for card in pool
            if card.suit is Suit.SPADES and card.rank.value < Rank.QUEEN.value
        ]
        if low_spades and view.opponent_has_suit[Suit.SPADES.value - 1]:
            return _highest_of(low_spades)

    if view.current_trick >= _LATE_TRICK:
        outs = [
            card
            for card in pool
            if _is_out_card(view, card) and not _later_void(view, card.suit)
        ]
        if outs:
            return _lowest_of(outs)

    for suit in (Suit.DIAMONDS, Suit.CLUBS):
        if _later_void(view, suit):
            continue
        suited = [card for card in pool if card.suit is suit]
        if suited:
            return _suit_by_freshness(view, suited)

    if view.qs_played:
        spades = [card for card in pool if card.suit is Suit.SPADES]
        if spades and not _later_void(view, Suit.SPADES):
            return _lowest_of(spades)

    if view.hearts_broken:
        hearts = [card for card in pool if card.is_heart]
        low_hearts = [card for card in hearts if card.rank.value <= _LOW_MAX.value]
        if (
            len(hearts) >= 3
            and len(low_hearts) >= 2
            and not _later_void(view, Suit.HEARTS)
        ):
            return _lowest_of(low_hearts)

    return _safe_lead_from(view, pool)


def _lead_to_void(view: WorldView, pool: list[Card]) -> Card | None:
    """Holding QS: lead a short clubs/diamonds suit to build a dump void."""
    cover = sum(
        1
        for card in view.hand
        if card.suit is Suit.SPADES and card.rank.value < Rank.QUEEN.value
    )
    by_len: list[tuple[int, Suit]] = []
    for suit in (Suit.DIAMONDS, Suit.CLUBS):
        n = sum(1 for card in view.hand if card.suit is suit)
        if n:
            by_len.append((n, suit))
    if cover >= _QS_COVER:
        n_spades = sum(1 for card in view.hand if card.suit is Suit.SPADES)
        if n_spades:
            by_len.append((n_spades, Suit.SPADES))
    by_len.sort(key=lambda item: (item[0], _TIE_SUIT[item[1]]))
    for _, suit in by_len:
        if _later_void(view, suit):
            continue
        suited = [card for card in pool if card.suit is suit]
        if suited:
            return _safe_lead_from(view, suited)
    non_spade = [
        card
        for card in pool
        if card.suit is not Suit.SPADES and not card.is_queen_of_spades
    ]
    if non_spade:
        return _safe_lead_from(view, non_spade)
    return None


def _safe_lead_from(view: WorldView, cards: list[Card]) -> Card:
    if not cards:
        raise ValueError("empty lead set")
    known_void = any(view.shown_voids[seat] for seat in range(1, NUM_PLAYERS))
    pool = cards
    if known_void:
        non_aces = [card for card in cards if card.rank is not Rank.ACE]
        if non_aces:
            pool = non_aces
    preferred = [card for card in pool if not _later_void(view, card.suit)]
    return _lowest_of(preferred or pool)


def _suit_by_freshness(view: WorldView, cards: list[Card]) -> Card:
    """Highest on the first two leads of a suit if nobody is known void."""
    suit = cards[0].suit
    if _early_safe_suit(view, suit):
        return _highest_of(cards)
    return _safe_lead_from(view, cards)


def _early_safe_suit(view: WorldView, suit: Suit) -> bool:
    """First two club/diamond leads with no demonstrated void in that suit."""
    if suit not in (Suit.CLUBS, Suit.DIAMONDS):
        return False
    count = view.suit_lead_counts[suit.value - 1]
    if view.is_leading:
        early = count < 2
    else:
        early = count <= 2
    if not early:
        return False
    return not any(suit in view.shown_voids[seat] for seat in range(1, NUM_PLAYERS))


def _without_spade_danger(view: WorldView, legal: list[Card]) -> list[Card]:
    if view.qs_played:
        return list(legal)
    safe = [
        card
        for card in legal
        if not (
            card.is_queen_of_spades
            or (
                card.suit is Suit.SPADES
                and card.rank in (Rank.ACE, Rank.KING)
            )
        )
    ]
    return safe or list(legal)


# --- follow ---------------------------------------------------------------------


def _follow_card(view: WorldView, legal: list[Card]) -> Card:
    if view.suit_led is Suit.SPADES:
        return _follow_spades(view, legal)
    if view.suit_led in (Suit.CLUBS, Suit.DIAMONDS) and _early_safe_suit(
        view, view.suit_led
    ):
        return _highest_of(legal)

    under = _under(legal, view.winning_card)
    over = _over(legal, view.winning_card)

    if view.is_last and not view.trick_has_points and not view.trick_has_qs:
        return _highest_of(legal)
    if under:
        return _highest_of(under)
    if view.trick_has_points or view.trick_has_qs:
        return _lowest_of(over or legal)
    return _lowest_of(legal)


def _follow_spades(view: WorldView, legal: list[Card]) -> Card:
    winning = view.winning_card
    as_or_ks_winning = (
        winning is not None
        and winning.suit is Suit.SPADES
        and winning.rank in (Rank.ACE, Rank.KING)
    )
    qs = next((card for card in legal if card.is_queen_of_spades), None)

    if view.qs_in_hand:
        if qs is not None and (as_or_ks_winning or len(legal) == 1):
            return qs
        cover = [card for card in legal if not card.is_queen_of_spades]
        if cover:
            under = _under(cover, winning)
            if under:
                return _highest_of(under)
            return _lowest_of(cover)
        assert qs is not None
        return qs

    if view.is_last and not view.trick_has_qs:
        aks = [card for card in legal if card.rank in (Rank.ACE, Rank.KING)]
        if aks:
            return _highest_of(aks)

    pool = legal
    if not view.qs_played and not view.trick_has_qs:
        safe = [
            card
            for card in pool
            if card.rank not in (Rank.ACE, Rank.KING)
        ]
        if safe:
            pool = safe

    under = _under(pool, winning)
    over = _over(pool, winning)
    if view.trick_has_qs:
        if under:
            return _highest_of(under)
        return _lowest_of(pool)
    if view.is_last and not view.trick_has_points:
        return _highest_of(pool)
    if under:
        return _highest_of(under)
    return _lowest_of(over or pool)


# --- sluff ----------------------------------------------------------------------


def _sluff_card(view: WorldView, legal: list[Card]) -> Card:
    if view.is_first_trick:
        non_points = [
            card
            for card in legal
            if not card.is_heart and not card.is_queen_of_spades
        ]
        return _highest_of(non_points or legal)

    qs = next((card for card in legal if card.is_queen_of_spades), None)
    if qs is not None and view.winning_seat not in (None, 0):
        return qs

    if not view.qs_played:
        unguarded = _unguarded_high_spades(view, legal)
        if unguarded:
            return _highest_of(unguarded)

    hearts = [card for card in legal if card.is_heart]
    if hearts:
        return _highest_of(hearts)

    highs = [card for card in legal if card.rank.value >= Rank.KING.value]
    if highs:
        return _highest_of(highs)

    longest = _longest_suit(view.hand)
    in_long = [card for card in legal if card.suit is longest]
    if in_long:
        return _highest_of(in_long)
    return _highest_of(legal)


def _unguarded_high_spades(view: WorldView, legal: list[Card]) -> list[Card]:
    cover = sum(
        1
        for card in view.hand
        if card.suit is Suit.SPADES and card.rank.value < Rank.QUEEN.value
    )
    if cover >= _QS_COVER:
        return []
    return [
        card
        for card in legal
        if card.suit is Suit.SPADES and card.rank in (Rank.ACE, Rank.KING)
    ]


# --- card facts -----------------------------------------------------------------


def _lowest_of(cards: list[Card]) -> Card:
    return min(cards, key=lambda card: (card.rank.value, _TIE_SUIT[card.suit]))


def _highest_of(cards: list[Card]) -> Card:
    return max(cards, key=lambda card: (card.rank.value, _TIE_SUIT[card.suit]))


def _under(cards: list[Card], winning: Card | None) -> list[Card]:
    if winning is None:
        return []
    return [card for card in cards if card.rank.value < winning.rank.value]


def _over(cards: list[Card], winning: Card | None) -> list[Card]:
    if winning is None:
        return list(cards)
    return [card for card in cards if card.rank.value > winning.rank.value]


def _suit_len(hand: tuple[Card, ...], suit: Suit | None) -> int:
    if suit is None:
        return 0
    return sum(1 for card in hand if card.suit is suit)


def _longest_suit(hand: tuple[Card, ...]) -> Suit:
    return max(_SUITS, key=lambda suit: _suit_len(hand, suit))


def _later_relative_seats(view: WorldView) -> list[int]:
    leader_rel = (NUM_PLAYERS - view.my_role) % NUM_PLAYERS
    return [
        (leader_rel + pos) % NUM_PLAYERS
        for pos in range(view.my_role + 1, NUM_PLAYERS)
    ]


def _later_void(view: WorldView, suit: Suit) -> bool:
    return any(suit in view.voids[seat] for seat in _later_relative_seats(view))


def _is_out_card(view: WorldView, card: Card) -> bool:
    if not view.opponent_has_suit[card.suit_index]:
        return False
    lowest = _lowest_remaining_rank(view, card.suit)
    return lowest is not None and card.rank_index == lowest


def _lowest_remaining_rank(view: WorldView, suit: Suit) -> int | None:
    start = (suit.value - 1) * 13
    for rank_i in range(13):
        if (start + rank_i) not in view.played_ids:
            return rank_i
    return None
