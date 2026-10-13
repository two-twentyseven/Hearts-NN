from __future__ import annotations

import numpy as np
import torch

from hearts.cards import Card, Deck, Rank, Suit
from hearts.config import GameConfig
from hearts.encode import encode_state
from hearts.env import HoldRoundEnv
from hearts.human_bot import (
    decode_observation,
    play_heuristic_hold_round,
    policy_from_choice,
    select_card,
    select_pass,
)
from hearts.model import legal_mask_from_cards
from hearts.rules import NUM_PLAYERS
from hearts.state import DECK_SIZE, GameState, PassDirection, Phase
from test_encode import _mid_round_state


def c(rank: Rank, suit: Suit) -> Card:
    return Card(rank, suit)


def _playing_state(hands: list[list[Card]], leader: int = 0) -> GameState:
    return GameState(
        config=GameConfig(),
        phase=Phase.PLAYING,
        round_number=1,
        hands=hands,
        pass_direction=PassDirection.HOLD,
        trick_number=1,
        trick_leader=leader,
        current_player=leader,
        hearts_broken=False,
        scores=[0, 0, 0, 0],
        tricks_won=[[] for _ in range(NUM_PLAYERS)],
    )


def _play(state: GameState, *cards: Card) -> GameState:
    for card in cards:
        state = state.play_card(state.current_player, card)
    return state


def _bot_choice(state: GameState) -> Card:
    legal = state.legal_moves_for()
    assert legal
    obs = encode_state(state, state.current_player)
    return select_card(obs, legal_mask_from_cards(legal))


def test_first_trick_followers_play_highest_club() -> None:
    two_clubs = c(Rank.TWO, Suit.CLUBS)
    three_clubs = c(Rank.THREE, Suit.CLUBS)
    king_clubs = c(Rank.KING, Suit.CLUBS)
    four_clubs = c(Rank.FOUR, Suit.CLUBS)
    ace_clubs = c(Rank.ACE, Suit.CLUBS)
    five_clubs = c(Rank.FIVE, Suit.CLUBS)
    nine_clubs = c(Rank.NINE, Suit.CLUBS)
    state = _playing_state(
        [
            [two_clubs, c(Rank.TWO, Suit.DIAMONDS), c(Rank.TWO, Suit.HEARTS)],
            [three_clubs, king_clubs, c(Rank.THREE, Suit.DIAMONDS)],
            [four_clubs, ace_clubs, c(Rank.FOUR, Suit.DIAMONDS)],
            [five_clubs, nine_clubs, c(Rank.FIVE, Suit.DIAMONDS)],
        ],
        leader=0,
    )
    state = _play(state, two_clubs)
    assert _bot_choice(state) == king_clubs
    state = _play(state, king_clubs)
    assert _bot_choice(state) == ace_clubs
    state = _play(state, ace_clubs)
    assert _bot_choice(state) == nine_clubs


def test_early_diamond_followers_play_highest_when_no_void() -> None:
    two_clubs = c(Rank.TWO, Suit.CLUBS)
    ace_clubs = c(Rank.ACE, Suit.CLUBS)
    four_diamonds = c(Rank.FOUR, Suit.DIAMONDS)
    ace_diamonds = c(Rank.ACE, Suit.DIAMONDS)
    eight_diamonds = c(Rank.EIGHT, Suit.DIAMONDS)
    five_diamonds = c(Rank.FIVE, Suit.DIAMONDS)
    state = _playing_state(
        [
            [two_clubs, four_diamonds, ace_diamonds],
            [c(Rank.THREE, Suit.CLUBS), c(Rank.SIX, Suit.DIAMONDS), c(Rank.TWO, Suit.HEARTS)],
            [ace_clubs, eight_diamonds, c(Rank.THREE, Suit.HEARTS)],
            [c(Rank.FOUR, Suit.CLUBS), five_diamonds, c(Rank.FOUR, Suit.HEARTS)],
        ],
        leader=0,
    )
    state = _play(
        state,
        two_clubs,
        c(Rank.THREE, Suit.CLUBS),
        ace_clubs,
        c(Rank.FOUR, Suit.CLUBS),
        eight_diamonds,
        five_diamonds,
    )
    assert state.current_player == 0
    choice = _bot_choice(state)
    assert choice == ace_diamonds
    assert choice != four_diamonds


def test_follow_ducks_when_someone_is_void_in_the_suit() -> None:
    two_clubs = c(Rank.TWO, Suit.CLUBS)
    ace_clubs = c(Rank.ACE, Suit.CLUBS)
    four_diamonds = c(Rank.FOUR, Suit.DIAMONDS)
    ace_diamonds = c(Rank.ACE, Suit.DIAMONDS)
    eight_diamonds = c(Rank.EIGHT, Suit.DIAMONDS)
    two_spades = c(Rank.TWO, Suit.SPADES)
    state = _playing_state(
        [
            [two_clubs, four_diamonds, ace_diamonds],
            [c(Rank.THREE, Suit.CLUBS), c(Rank.SIX, Suit.DIAMONDS), c(Rank.TWO, Suit.HEARTS)],
            [ace_clubs, eight_diamonds, c(Rank.THREE, Suit.HEARTS)],
            [c(Rank.FOUR, Suit.CLUBS), two_spades, c(Rank.FOUR, Suit.HEARTS)],
        ],
        leader=0,
    )
    # Trick 1 clubs, P2 wins. Trick 2: P2 leads 8D, P3 sluffs (void diamonds).
    state = _play(
        state,
        two_clubs,
        c(Rank.THREE, Suit.CLUBS),
        ace_clubs,
        c(Rank.FOUR, Suit.CLUBS),
        eight_diamonds,
        two_spades,
    )
    assert state.current_player == 0
    choice = _bot_choice(state)
    assert choice == four_diamonds
    assert choice != ace_diamonds


def test_decode_mid_round_fixture() -> None:
    state = _mid_round_state()
    obs = encode_state(state, acting_player=0)
    view = decode_observation(obs)

    assert view.hand == (c(Rank.TWO, Suit.HEARTS),)
    assert view.hearts_broken
    assert view.current_trick == 2
    assert view.suit_led is Suit.CLUBS
    assert view.points_taken[0] == 1
    assert all(pts == 0 for pts in view.points_taken[1:])
    assert Suit.SPADES in view.voids[3]
    played = {play.card for play in view.current_plays}
    assert played == {c(Rank.FIVE, Suit.CLUBS), c(Rank.SIX, Suit.CLUBS)}


def test_full_round_stays_legal_and_terminates() -> None:
    for seed in (0, 1, 7, 21, 99):
        env = HoldRoundEnv()
        state = env.reset(seed=seed)
        steps = 0
        while not env.is_terminal:
            legal = state.legal_moves_for()
            assert legal
            card = _bot_choice(state)
            assert card in legal
            state = env.step(card)
            steps += 1
        assert steps == 52
        assert env.round_scores is not None
        assert len(env.round_scores) == NUM_PLAYERS
        assert sum(env.round_scores) in (26, 78)  # normal, or moon (26*3)


def test_sluff_dumps_qs_when_someone_else_is_winning() -> None:
    two_clubs = c(Rank.TWO, Suit.CLUBS)
    qs = c(Rank.QUEEN, Suit.SPADES)
    two_hearts = c(Rank.TWO, Suit.HEARTS)
    king_clubs = c(Rank.KING, Suit.CLUBS)
    three_clubs = c(Rank.THREE, Suit.CLUBS)
    four_clubs = c(Rank.FOUR, Suit.CLUBS)
    state = _playing_state(
        [
            [two_clubs, qs, two_hearts],
            [king_clubs, c(Rank.THREE, Suit.DIAMONDS), c(Rank.FOUR, Suit.DIAMONDS)],
            [three_clubs, c(Rank.FIVE, Suit.DIAMONDS), c(Rank.SIX, Suit.DIAMONDS)],
            [four_clubs, c(Rank.SEVEN, Suit.DIAMONDS), c(Rank.EIGHT, Suit.DIAMONDS)],
        ],
        leader=0,
    )
    state = _play(
        state,
        two_clubs,
        king_clubs,
        three_clubs,
        four_clubs,
        c(Rank.THREE, Suit.DIAMONDS),
        c(Rank.FIVE, Suit.DIAMONDS),
        c(Rank.SEVEN, Suit.DIAMONDS),
    )
    assert state.current_player == 0
    choice = _bot_choice(state)
    assert choice == qs
    assert choice != two_hearts


def test_lead_low_spade_not_ace_when_queen_is_out() -> None:
    ace_clubs = c(Rank.ACE, Suit.CLUBS)
    two_spades = c(Rank.TWO, Suit.SPADES)
    three_spades = c(Rank.THREE, Suit.SPADES)
    ace_spades = c(Rank.ACE, Suit.SPADES)
    four_diamonds = c(Rank.FOUR, Suit.DIAMONDS)
    two_clubs = c(Rank.TWO, Suit.CLUBS)
    state = _playing_state(
        [
            [ace_clubs, two_spades, three_spades, ace_spades, four_diamonds],
            [two_clubs, c(Rank.FOUR, Suit.CLUBS), c(Rank.FIVE, Suit.DIAMONDS)],
            [c(Rank.THREE, Suit.CLUBS), c(Rank.FIVE, Suit.CLUBS), c(Rank.SIX, Suit.DIAMONDS)],
            [c(Rank.SIX, Suit.CLUBS), c(Rank.SEVEN, Suit.CLUBS), c(Rank.EIGHT, Suit.DIAMONDS)],
        ],
        leader=1,
    )
    state = _play(
        state,
        two_clubs,
        c(Rank.THREE, Suit.CLUBS),
        c(Rank.SIX, Suit.CLUBS),
        ace_clubs,
    )
    assert state.current_player == 0
    choice = _bot_choice(state)
    assert choice == three_spades
    assert choice != ace_spades


def test_holding_qs_does_not_lead_high_spades() -> None:
    ace_clubs = c(Rank.ACE, Suit.CLUBS)
    qs = c(Rank.QUEEN, Suit.SPADES)
    ace_spades = c(Rank.ACE, Suit.SPADES)
    king_spades = c(Rank.KING, Suit.SPADES)
    four_diamonds = c(Rank.FOUR, Suit.DIAMONDS)
    two_clubs = c(Rank.TWO, Suit.CLUBS)
    state = _playing_state(
        [
            [ace_clubs, qs, ace_spades, king_spades, four_diamonds],
            [two_clubs, c(Rank.FOUR, Suit.CLUBS), c(Rank.FIVE, Suit.DIAMONDS)],
            [c(Rank.THREE, Suit.CLUBS), c(Rank.FIVE, Suit.CLUBS), c(Rank.SIX, Suit.DIAMONDS)],
            [c(Rank.SIX, Suit.CLUBS), c(Rank.SEVEN, Suit.CLUBS), c(Rank.EIGHT, Suit.DIAMONDS)],
        ],
        leader=1,
    )
    state = _play(
        state,
        two_clubs,
        c(Rank.THREE, Suit.CLUBS),
        c(Rank.SIX, Suit.CLUBS),
        ace_clubs,
    )
    assert state.current_player == 0
    choice = _bot_choice(state)
    assert choice == four_diamonds
    assert choice not in {qs, ace_spades, king_spades}


def test_moon_stop_takes_a_heart_when_one_opponent_has_all_points() -> None:
    two_clubs = c(Rank.TWO, Suit.CLUBS)
    four_hearts = c(Rank.FOUR, Suit.HEARTS)
    ace_hearts = c(Rank.ACE, Suit.HEARTS)
    two_diamonds = c(Rank.TWO, Suit.DIAMONDS)
    king_clubs = c(Rank.KING, Suit.CLUBS)
    king_hearts = c(Rank.KING, Suit.HEARTS)
    ace_diamonds = c(Rank.ACE, Suit.DIAMONDS)
    three_spades = c(Rank.THREE, Suit.SPADES)
    three_clubs = c(Rank.THREE, Suit.CLUBS)
    three_hearts = c(Rank.THREE, Suit.HEARTS)
    four_diamonds = c(Rank.FOUR, Suit.DIAMONDS)
    four_clubs = c(Rank.FOUR, Suit.CLUBS)
    five_hearts = c(Rank.FIVE, Suit.HEARTS)
    six_hearts = c(Rank.SIX, Suit.HEARTS)
    state = _playing_state(
        [
            [two_clubs, four_hearts, ace_hearts, two_diamonds],
            [king_clubs, king_hearts, ace_diamonds, three_spades],
            [three_clubs, three_hearts, four_diamonds],
            [four_clubs, five_hearts, six_hearts],
        ],
        leader=0,
    )
    # Trick 1: 2C-KC-3C-4C, player 1 wins.
    # Trick 2: AD-4D-5H-2D, player 1 takes the sluffed heart.
    # Trick 3: KH-3H-6H, player 0 can duck 4H or take with AH.
    state = _play(
        state,
        two_clubs,
        king_clubs,
        three_clubs,
        four_clubs,
        ace_diamonds,
        four_diamonds,
        five_hearts,
        two_diamonds,
        king_hearts,
        three_hearts,
        six_hearts,
    )
    assert state.current_player == 0
    assert state.last_round_scores is None
    obs = encode_state(state, 0)
    view = decode_observation(obs)
    assert view.points_taken[1] > 0
    assert view.points_taken[0] == 0
    choice = _bot_choice(state)
    assert choice == ace_hearts
    assert choice != four_hearts


def test_play_heuristic_hold_round_batch_shapes() -> None:
    batch = play_heuristic_hold_round(seed=5)
    n = batch["z"].shape[0]
    assert n == NUM_PLAYERS * 13
    assert batch["pi_target"].shape == (n, DECK_SIZE)
    assert batch["legal_mask"].shape == batch["pi_target"].shape
    assert torch.allclose(batch["pi_target"].sum(dim=1), torch.ones(n))
    chosen = batch["pi_target"].argmax(dim=1)
    assert torch.all(batch["legal_mask"][torch.arange(n), chosen])
    assert torch.all(batch["pi_target"].max(dim=1).values == 1.0)
    assert batch["acting_player"].shape == (n,)
    assert batch["z"].ndim == 1


def test_policy_from_choice_is_one_hot() -> None:
    card = c(Rank.QUEEN, Suit.SPADES)
    pi = policy_from_choice(card)
    assert pi.shape == (DECK_SIZE,)
    assert float(pi.sum()) == 1.0
    assert float(pi[card.card_id]) == 1.0


def test_select_card_is_deterministic() -> None:
    env = HoldRoundEnv()
    state = env.reset(seed=3)
    obs = encode_state(state, state.current_player)
    mask = legal_mask_from_cards(state.legal_moves_for())
    assert select_card(obs, mask) == select_card(obs, mask)
    assert select_card(obs, mask) == select_card(obs, np.asarray(mask))


def _pass_hand(*cards: Card, fill: Suit = Suit.CLUBS) -> list[Card]:
    """Pad to 13 cards with low non-spade cards, skipping anything already held."""
    held = set(cards)
    extras: list[Card] = []

    def add_from(suit: Suit) -> None:
        for rank in Rank:
            if len(cards) + len(extras) >= 13:
                return
            if rank.value > Rank.SEVEN.value:
                continue
            card = c(rank, suit)
            if card not in held and card not in extras:
                extras.append(card)

    add_from(fill)
    for suit in (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS):
        if suit is not fill:
            add_from(suit)
    add_from(Suit.SPADES)
    hand = list(cards) + extras
    assert len(hand) == 13
    return hand


def test_pass_unprotected_ace_and_king_of_spades() -> None:
    ace = c(Rank.ACE, Suit.SPADES)
    king = c(Rank.KING, Suit.SPADES)
    two_spades = c(Rank.TWO, Suit.SPADES)
    passed = select_pass(_pass_hand(ace, king, two_spades, fill=Suit.DIAMONDS))
    assert ace in passed
    assert king in passed
    assert two_spades not in passed


def test_pass_keeps_protected_ace_of_spades() -> None:
    ace = c(Rank.ACE, Suit.SPADES)
    cover = [
        c(Rank.TWO, Suit.SPADES),
        c(Rank.THREE, Suit.SPADES),
        c(Rank.FOUR, Suit.SPADES),
    ]
    passed = select_pass(_pass_hand(ace, *cover, fill=Suit.DIAMONDS))
    assert ace not in passed
    assert all(card not in passed for card in cover)


def test_pass_unprotected_queen_of_spades() -> None:
    queen = c(Rank.QUEEN, Suit.SPADES)
    two_spades = c(Rank.TWO, Suit.SPADES)
    three_spades = c(Rank.THREE, Suit.SPADES)
    passed = select_pass(_pass_hand(queen, two_spades, three_spades, fill=Suit.DIAMONDS))
    assert queen in passed
    assert two_spades not in passed
    assert three_spades not in passed


def test_pass_keeps_queen_with_three_other_spades() -> None:
    queen = c(Rank.QUEEN, Suit.SPADES)
    cover = [
        c(Rank.TWO, Suit.SPADES),
        c(Rank.THREE, Suit.SPADES),
        c(Rank.FOUR, Suit.SPADES),
    ]
    passed = select_pass(_pass_hand(queen, *cover, fill=Suit.DIAMONDS))
    assert queen not in passed
    assert all(card not in passed for card in cover)


def test_pass_dumps_queen_after_unguarded_ak() -> None:
    queen = c(Rank.QUEEN, Suit.SPADES)
    ace = c(Rank.ACE, Suit.SPADES)
    king = c(Rank.KING, Suit.SPADES)
    two_spades = c(Rank.TWO, Suit.SPADES)
    passed = select_pass(_pass_hand(queen, ace, king, two_spades, fill=Suit.DIAMONDS))
    assert set(passed) == {queen, ace, king}


def test_pass_never_chooses_low_spades_when_off_suit_exists() -> None:
    spades = [
        c(Rank.TWO, Suit.SPADES),
        c(Rank.THREE, Suit.SPADES),
        c(Rank.FOUR, Suit.SPADES),
        c(Rank.FIVE, Suit.SPADES),
        c(Rank.SIX, Suit.SPADES),
        c(Rank.SEVEN, Suit.SPADES),
        c(Rank.EIGHT, Suit.SPADES),
        c(Rank.NINE, Suit.SPADES),
        c(Rank.JACK, Suit.SPADES),
    ]
    off = [
        c(Rank.TWO, Suit.HEARTS),
        c(Rank.THREE, Suit.HEARTS),
        c(Rank.FOUR, Suit.CLUBS),
        c(Rank.FIVE, Suit.DIAMONDS),
    ]
    passed = select_pass(spades + off)
    assert all(card.suit is not Suit.SPADES for card in passed)
    assert set(passed) <= set(off)


def test_pass_danger_suit_prefers_higher_average_cost() -> None:
    ace_diamonds = c(Rank.ACE, Suit.DIAMONDS)
    king_diamonds = c(Rank.KING, Suit.DIAMONDS)
    two_diamonds = c(Rank.TWO, Suit.DIAMONDS)
    ace_hearts = c(Rank.ACE, Suit.HEARTS)
    eight_hearts = c(Rank.EIGHT, Suit.HEARTS)
    two_hearts = c(Rank.TWO, Suit.HEARTS)
    three_hearts = c(Rank.THREE, Suit.HEARTS)
    # Diamonds avg (14+13+2)/3 = 9.67; hearts (14+8+2+3)/4 = 6.75.
    passed = select_pass(
        _pass_hand(
            ace_diamonds,
            king_diamonds,
            two_diamonds,
            ace_hearts,
            eight_hearts,
            two_hearts,
            three_hearts,
            fill=Suit.CLUBS,
        )
    )
    assert passed[0] == ace_diamonds
    assert passed[1] == king_diamonds
    assert passed[2] == ace_hearts
    assert two_diamonds not in passed
    assert eight_hearts not in passed


def test_pass_safe_suit_with_three_buffers_is_not_dumped_first() -> None:
    ace_hearts = c(Rank.ACE, Suit.HEARTS)
    hearts = [
        ace_hearts,
        c(Rank.TWO, Suit.HEARTS),
        c(Rank.THREE, Suit.HEARTS),
        c(Rank.FOUR, Suit.HEARTS),
    ]
    ace_clubs = c(Rank.ACE, Suit.CLUBS)
    eight_clubs = c(Rank.EIGHT, Suit.CLUBS)
    passed = select_pass(
        _pass_hand(ace_clubs, eight_clubs, *hearts, fill=Suit.DIAMONDS)
    )
    assert ace_clubs in passed
    assert eight_clubs in passed
    assert ace_hearts not in passed


def test_pass_dumps_high_hearts_instead_of_shooting_moon() -> None:
    hearts = [
        c(Rank.ACE, Suit.HEARTS),
        c(Rank.KING, Suit.HEARTS),
        c(Rank.QUEEN, Suit.HEARTS),
        c(Rank.JACK, Suit.HEARTS),
        c(Rank.TEN, Suit.HEARTS),
        c(Rank.NINE, Suit.HEARTS),
        c(Rank.EIGHT, Suit.HEARTS),
    ]
    spade_cover = [
        c(Rank.TWO, Suit.SPADES),
        c(Rank.THREE, Suit.SPADES),
        c(Rank.FOUR, Suit.SPADES),
        c(Rank.FIVE, Suit.SPADES),
    ]
    passed = select_pass(
        _pass_hand(*hearts, *spade_cover, fill=Suit.CLUBS)
    )
    assert set(passed) == {
        c(Rank.ACE, Suit.HEARTS),
        c(Rank.KING, Suit.HEARTS),
        c(Rank.QUEEN, Suit.HEARTS),
    }


def test_select_pass_always_returns_three_from_hand() -> None:
    for seed in range(20):
        hand = Deck.fresh(seed).deal()[0]
        passed = select_pass(hand)
        assert len(passed) == 3
        assert len(set(passed)) == 3
        assert all(card in hand for card in passed)
        assert select_pass(hand) == passed
