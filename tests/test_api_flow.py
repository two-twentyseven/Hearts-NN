from __future__ import annotations

from hearts.api.app import (
    CreateGameRequest,
    PassRequest,
    PlayRequest,
    create_game,
    play_card,
    submit_pass,
    tick_bot,
)
from hearts.api.bot import bot_pass_cards
from hearts.cards import Card, Rank, Suit


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


def test_create_game_defaults_to_random_difficulty() -> None:
    state = create_game()
    assert state["difficulty"] == "random"


def test_create_game_accepts_human_difficulty() -> None:
    state = create_game(CreateGameRequest(difficulty="human"))
    assert state["difficulty"] == "human"
    game_id = state["game_id"]
    state = submit_pass(game_id, PassRequest(cards=state["hand"][:3]))
    assert state["phase"] == "PLAYING"
    assert state["difficulty"] == "human"

    if state["current_player"] == state["viewer"]:
        state = play_card(game_id, PlayRequest(card=state["legal_moves"][0]))
    next_state = tick_bot(game_id)
    assert next_state["action"]["type"] == "play"


def test_human_bot_pass_is_not_random_sample() -> None:
    ace = Card(Rank.ACE, Suit.SPADES)
    king = Card(Rank.KING, Suit.SPADES)
    two_spades = Card(Rank.TWO, Suit.SPADES)
    hand = [
        ace,
        king,
        two_spades,
        *[Card(rank, Suit.DIAMONDS) for rank in Rank if rank.value <= 7],
        Card(Rank.TWO, Suit.CLUBS),
        Card(Rank.THREE, Suit.CLUBS),
        Card(Rank.FOUR, Suit.CLUBS),
        Card(Rank.FIVE, Suit.CLUBS),
    ]
    assert len(hand) == 13
    passed = bot_pass_cards(hand, "human")
    assert ace in passed
    assert king in passed
    assert two_spades not in passed
