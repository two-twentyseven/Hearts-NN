from __future__ import annotations

from hearts.api.app import (
    CreateGameRequest,
    PassRequest,
    PlayRequest,
    create_game,
    next_round,
    play_card,
    submit_pass,
    tick_bot,
)
from hearts.api.serialize import state_for_viewer
from hearts.state import Phase


def _pass_and_start(difficulty: str = "random") -> dict:
    state = create_game(CreateGameRequest(difficulty=difficulty))
    assert state["hands"][0] == state["hand"]
    assert all(len(hand) == 13 for hand in state["hands"])
    assert state["replay"]["plays"] == []
    assert state["replay"]["opening"] is None
    return submit_ready(state)


def submit_ready(state: dict) -> dict:
    return submit_pass(state["game_id"], PassRequest(cards=state["hand"][:3]))


def _advance_one(state: dict) -> dict:
    game_id = state["game_id"]
    if state["current_player"] == state["viewer"] and state["phase"] == "PLAYING":
        return play_card(game_id, PlayRequest(card=state["legal_moves"][0]))
    return tick_bot(game_id)


def test_reveal_hands_are_present_in_every_difficulty() -> None:
    for difficulty in ("random", "human"):
        state = create_game(CreateGameRequest(difficulty=difficulty))
        assert len(state["hands"]) == 4
        assert state["hands"][state["viewer"]] == state["hand"]


def test_each_play_keeps_the_hand_the_card_was_chosen_from() -> None:
    state = _pass_and_start()
    opening = state["replay"]["opening"]
    assert opening["kind"] == "opening"
    assert opening["card"] is None
    assert all(len(hand) == 13 for hand in opening["hands"])
    assert state["replay"]["plays"] == []

    state = _advance_one(state)
    play = state["replay"]["plays"][-1]
    actor = play["actor"]
    card = play["card"]

    assert play["kind"] == "play"
    assert card in play["decision_hand"]
    assert card in play["legal_moves"]
    assert card not in play["hands"][actor]
    assert any(step["player"] == actor and step["card"] == card for step in play["current_trick"])
    assert len(play["decision_hand"]) == len(play["hands"][actor]) + 1


def test_completed_trick_stays_on_the_table_in_that_frame() -> None:
    state = _pass_and_start()
    for _ in range(4):
        state = _advance_one(state)
    play = state["replay"]["plays"][-1]
    assert len(play["current_trick"]) == 4
    assert play["trick_winner"] is not None
    assert sum(play["tricks_won_counts"]) == 0
    assert state["replay"]["opening"]["hands"][0]


def test_next_round_clears_the_review_log() -> None:
    state = _pass_and_start()
    while state["phase"] == "PLAYING":
        state = _advance_one(state)
    assert state["phase"] == "ROUND_OVER"
    assert len(state["replay"]["plays"]) == 52

    state = next_round(state["game_id"])
    assert state["replay"]["plays"] == []
    if state["phase"] == "PLAYING":
        assert state["replay"]["opening"] is not None
    else:
        assert state["phase"] == "PASSING"
        assert state["replay"]["opening"] is None


def test_viewer_serialization_includes_every_seat() -> None:
    from hearts.api.game_store import store

    state = create_game()
    live = store.get(state["game_id"])
    assert live is not None
    view = state_for_viewer(live, store.human_player, game_id=state["game_id"])
    assert len(view["hands"]) == 4
    assert live.phase is Phase.PASSING
