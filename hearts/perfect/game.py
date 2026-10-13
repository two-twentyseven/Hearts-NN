from __future__ import annotations

from dataclasses import replace

from hearts.cards import Card, Deck, Rank, Suit
from hearts.config import GameConfig
from hearts.rules import CARDS_PER_HAND, NUM_PLAYERS, PassDirection, two_of_clubs_holder
from hearts.state import DECK_SIZE, GameState, Phase, _blank_card_play_tricks, _blank_trick_leaders

_SUITS = list(Suit)
_RANKS = list(Rank)
_CARDS_BY_ID: list[Card] = [Card(_RANKS[i % 13], _SUITS[i // 13]) for i in range(DECK_SIZE)]


def card_from_id(card_id: int) -> Card:
    """Rebuild a Card from its canonical 0-51 id."""
    return _CARDS_BY_ID[card_id]


def new_hold_round(seed: int | None = None, config: GameConfig | None = None) -> GameState:
    """Deal a fresh hold round (no passing) and return the state at the first lead."""
    cfg = config or GameConfig()
    if seed is not None:
        cfg = replace(cfg, seed=seed)
    deck = Deck.fresh(cfg.seed)
    hands = deck.deal(NUM_PLAYERS, CARDS_PER_HAND)
    leader = two_of_clubs_holder(hands)
    leaders = _blank_trick_leaders()
    leaders[0] = leader
    return GameState(
        config=cfg,
        phase=Phase.PLAYING,
        round_number=1,
        hands=hands,
        pass_direction=PassDirection.HOLD,
        pending_passes={},
        trick_number=1,
        trick_leader=leader,
        current_player=leader,
        current_trick=[],
        hearts_broken=False,
        tricks_won=[[] for _ in range(NUM_PLAYERS)],
        scores=[0] * NUM_PLAYERS,
        last_round_scores=None,
        history=[],
        card_play_tricks=_blank_card_play_tricks(),
        trick_leaders=leaders,
    )


def strip_history(state: GameState) -> GameState:
    """Drop the event log so search copies stay cheap."""
    if not state.history:
        return state
    return replace(state, history=[])


def is_round_over(state: GameState) -> bool:
    return state.phase in (Phase.ROUND_OVER, Phase.GAME_OVER)
