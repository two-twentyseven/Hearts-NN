from __future__ import annotations

from typing import Any

from hearts.cards import Card, Rank, Suit
from hearts.rules import NUM_PLAYERS, PassDirection
from hearts.state import GameState, Phase

RANK_TO_CHAR: dict[Rank, str] = {
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "T",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
}

CHAR_TO_RANK: dict[str, Rank] = {value: key for key, value in RANK_TO_CHAR.items()}

SUIT_TO_CHAR: dict[Suit, str] = {
    Suit.CLUBS: "C",
    Suit.DIAMONDS: "D",
    Suit.HEARTS: "H",
    Suit.SPADES: "S",
}

CHAR_TO_SUIT: dict[str, Suit] = {value: key for key, value in SUIT_TO_CHAR.items()}

PASS_DIRECTION_LABELS: dict[PassDirection, str] = {
    PassDirection.LEFT: "LEFT",
    PassDirection.RIGHT: "RIGHT",
    PassDirection.ACROSS: "ACROSS",
    PassDirection.HOLD: "HOLD",
}

PASS_DIRECTION_MESSAGES: dict[PassDirection, str] = {
    PassDirection.LEFT: "Pass 3 cards to the left",
    PassDirection.RIGHT: "Pass 3 cards to the right",
    PassDirection.ACROSS: "Pass 3 cards across",
    PassDirection.HOLD: "No passing this round",
}

PLAYER_NAMES = ["You", "West", "North", "East"]


def card_to_id(card: Card) -> str:
    return f"{RANK_TO_CHAR[card.rank]}{SUIT_TO_CHAR[card.suit]}"


def card_from_id(card_id: str) -> Card:
    if len(card_id) != 2:
        raise ValueError(f"Invalid card id: {card_id}")
    rank_char, suit_char = card_id[0], card_id[1]
    if rank_char not in CHAR_TO_RANK or suit_char not in CHAR_TO_SUIT:
        raise ValueError(f"Invalid card id: {card_id}")
    return Card(CHAR_TO_RANK[rank_char], CHAR_TO_SUIT[suit_char])


def cards_from_ids(card_ids: list[str]) -> list[Card]:
    return [card_from_id(card_id) for card_id in card_ids]


def sort_cards(cards: list[Card]) -> list[Card]:
    return sorted(cards, key=lambda card: (card.suit.value, card.rank.value))


def _pass_submitted(state: GameState) -> list[bool]:
    return [player in state.pending_passes for player in range(NUM_PLAYERS)]


def _build_messages(state: GameState, viewer: int) -> list[str]:
    messages: list[str] = []

    if state.phase is Phase.PASSING:
        if viewer not in state.pending_passes:
            messages.append(PASS_DIRECTION_MESSAGES[state.pass_direction])
        else:
            messages.append("Waiting for other players to pass")
    elif state.phase is Phase.PLAYING:
        if state.current_player == viewer:
            messages.append("Your turn")
        else:
            messages.append(f"{PLAYER_NAMES[state.current_player]} is playing")
        if state.hearts_broken:
            messages.append("Hearts broken")
    elif state.phase is Phase.ROUND_OVER:
        messages.append("Round complete")
    elif state.phase is Phase.GAME_OVER:
        winner = min(range(NUM_PLAYERS), key=lambda player: state.scores[player])
        messages.append(f"{PLAYER_NAMES[winner]} wins!")

    return messages


def state_for_viewer(
    state: GameState,
    viewer: int,
    *,
    game_id: str,
) -> dict[str, Any]:
    hand = sort_cards(state.hands[viewer])
    legal_moves: list[Card] = []
    if state.phase is Phase.PLAYING and state.current_player == viewer:
        legal_moves = sort_cards(state.legal_moves_for(viewer))
    elif state.phase is Phase.PASSING and viewer not in state.pending_passes:
        legal_moves = hand

    return {
        "game_id": game_id,
        "phase": state.phase.name,
        "round_number": state.round_number,
        "pass_direction": PASS_DIRECTION_LABELS[state.pass_direction],
        "viewer": viewer,
        "hand": [card_to_id(card) for card in hand],
        "legal_moves": [card_to_id(card) for card in legal_moves],
        "hand_counts": [len(hand_cards) for hand_cards in state.hands],
        "current_player": state.current_player,
        "trick_leader": state.trick_leader,
        "trick_number": state.trick_number,
        "current_trick": [
            {"player": player, "card": card_to_id(card)}
            for player, card in state.current_trick
        ],
        "tricks_won_counts": [
            len(cards) // NUM_PLAYERS for cards in state.tricks_won
        ],
        "hearts_broken": state.hearts_broken,
        "scores": list(state.scores),
        "last_round_scores": (
            list(state.last_round_scores)
            if state.last_round_scores is not None
            else None
        ),
        "pass_submitted": _pass_submitted(state),
        "players": list(PLAYER_NAMES),
        "messages": _build_messages(state, viewer),
        "target_score": state.config.target_score,
    }
