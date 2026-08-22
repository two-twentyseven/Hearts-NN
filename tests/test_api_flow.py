from __future__ import annotations

from hearts.api.app import (
    PassRequest,
    PlayRequest,
    create_game,
    play_card,
    submit_pass,
    tick_bot,
)


def test_pass_returns_animation_payload() -> None:
    state = create_game()
    state = submit_pass(
        state["game_id"],
        PassRequest(cards=state["hand"][:3]),
    )

    assert state["action"]["type"] == "pass"
    assert len(state["action"]["cards"]) == 3
    assert len(state["action"]["incoming"]) == 3
    assert state["phase"] == "PLAYING"


def test_bot_tick_advances_only_one_play() -> None:
    state = create_game()
    game_id = state["game_id"]
    state = submit_pass(game_id, PassRequest(cards=state["hand"][:3]))

    if state["current_player"] == state["viewer"]:
        state = play_card(
            game_id,
            PlayRequest(card=state["legal_moves"][0]),
        )

    assert state["phase"] == "PLAYING"
    assert state["current_player"] != state["viewer"]
    before_counts = state["hand_counts"]

    next_state = tick_bot(game_id)
    action = next_state["action"]

    assert action["type"] == "play"
    assert action["player"] == state["current_player"]
    assert next_state["hand_counts"][action["player"]] == before_counts[action["player"]] - 1
    assert sum(next_state["hand_counts"]) == sum(before_counts) - 1
