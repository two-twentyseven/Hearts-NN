from __future__ import annotations

import random

from hearts.env import HoldRoundEnv, card_from_id, normalize_round_return
from hearts.rules import CARDS_PER_HAND, NUM_PLAYERS
from hearts.state import Phase


def test_card_from_id_roundtrip() -> None:
    for card_id in (0, 12, 13, 51):
        card = card_from_id(card_id)
        assert card.card_id == card_id


def test_normalize_round_return() -> None:
    assert normalize_round_return(0) == 1.0
    assert normalize_round_return(26) == -1.0
    assert abs(normalize_round_return(13) - 0.0) < 1e-6


def test_hold_round_env_completes_with_random_play() -> None:
    env = HoldRoundEnv()
    state = env.reset(seed=7)
    assert state.phase is Phase.PLAYING
    assert state.pass_direction.name == "HOLD"
    assert sum(len(h) for h in state.hands) == NUM_PLAYERS * CARDS_PER_HAND

    rng = random.Random(0)
    while not env.is_terminal:
        legal = state.legal_moves_for()
        assert legal
        card = rng.choice(legal)
        state = env.step(card)

    assert env.round_scores is not None
    assert len(env.round_scores) == NUM_PLAYERS
    returns = env.returns_by_player()
    assert len(returns) == NUM_PLAYERS
    assert all(-1.0 <= z <= 1.0 for z in returns)
