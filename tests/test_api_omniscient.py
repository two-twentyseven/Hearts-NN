from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi import HTTPException

from hearts.api.app import (
    CreateGameRequest,
    PassRequest,
    PlayRequest,
    create_game,
    health,
    play_card,
    submit_pass,
    tick_bot,
)
from hearts.api.bot import bot_pass_cards, bot_play
from hearts.api.serialize import card_from_id
from hearts.perfect.agent import OmniscientBot, set_shared_bot
from hearts.perfect.game import new_hold_round
from hearts.perfect.model import NetConfig, PerfectNet, save_checkpoint
from hearts.state import GameState, Phase


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "latest.pt"
    save_checkpoint(path, PerfectNet(NetConfig(d_model=32, n_blocks=1)), {"iteration": 0})
    return path


@pytest.fixture
def omniscient(tiny_checkpoint: Path):
    bot = OmniscientBot(tiny_checkpoint, n_simulations=8, seed=0)
    set_shared_bot(bot)
    yield bot
    set_shared_bot(None)


@pytest.fixture
def no_checkpoint(tmp_path: Path):
    set_shared_bot(OmniscientBot(tmp_path / "missing.pt", n_simulations=8))
    yield
    set_shared_bot(None)


def test_omniscient_bot_plays_legal_cards_through_a_hold_round(omniscient: OmniscientBot) -> None:
    state = new_hold_round(seed=5)
    while state.phase is Phase.PLAYING:
        card = omniscient.play(state)
        assert card in state.legal_moves_for(state.current_player)
        state = state.play_card(state.current_player, card)
    assert state.last_round_scores is not None
    assert omniscient.iteration == 0


def test_omniscient_bot_ignores_history_and_uses_match_scores(omniscient: OmniscientBot) -> None:
    # Full match state (with passing + history) after passes are applied.
    state = GameState.new_game()
    for player in range(4):
        state = state.submit_pass(player, state.hands[player][:3])
    assert state.phase is Phase.PLAYING and state.history
    card = omniscient.play(state)
    assert card in state.legal_moves_for(state.current_player)
    # Same position with the history wiped plays the same card (deterministic search, greedy).
    from dataclasses import replace

    bot2 = OmniscientBot(omniscient.checkpoint, n_simulations=8, seed=0)
    assert bot2.play(replace(state, history=[])) == card


def test_bot_hot_reloads_newer_checkpoint(tiny_checkpoint: Path) -> None:
    import os

    bot = OmniscientBot(tiny_checkpoint, n_simulations=4, seed=0)
    assert bot.iteration == 0
    save_checkpoint(tiny_checkpoint, PerfectNet(NetConfig(d_model=32, n_blocks=1)), {"iteration": 7})
    os.utime(tiny_checkpoint, (time.time() + 10, time.time() + 10))  # force a distinct mtime
    assert bot.iteration == 7
    # A corrupt file mid-write keeps the previously loaded net.
    tiny_checkpoint.write_bytes(b"garbage")
    os.utime(tiny_checkpoint, (time.time() + 20, time.time() + 20))
    assert bot.iteration == 7
    assert bot.play(new_hold_round(seed=1)) is not None


def test_bot_dispatch_uses_omniscient(omniscient: OmniscientBot) -> None:
    state = new_hold_round(seed=2)
    card = bot_play(state, "omniscient")
    assert card in state.legal_moves_for(state.current_player)
    passed = bot_pass_cards(state.hands[0], "omniscient")
    assert len(passed) == 3 and all(c in state.hands[0] for c in passed)


def test_api_omniscient_game_flow(omniscient: OmniscientBot) -> None:
    assert health()["omniscient_available"] is True
    state = create_game(CreateGameRequest(difficulty="omniscient"))
    assert state["difficulty"] == "omniscient"
    game_id = state["game_id"]
    state = submit_pass(game_id, PassRequest(cards=state["hand"][:3]))
    assert state["phase"] == "PLAYING"
    if state["current_player"] == state["viewer"]:
        state = play_card(game_id, PlayRequest(card=state["legal_moves"][0]))
    ticked = tick_bot(game_id)
    assert ticked["action"]["type"] == "play"
    card_from_id(ticked["action"]["card"])  # valid id


def test_api_rejects_omniscient_without_checkpoint(no_checkpoint) -> None:
    assert health()["omniscient_available"] is False
    with pytest.raises(HTTPException) as exc:
        create_game(CreateGameRequest(difficulty="omniscient"))
    assert exc.value.status_code == 503
    assert exc.value.detail["reason"] == "NO_CHECKPOINT"
