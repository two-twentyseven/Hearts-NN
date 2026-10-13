from __future__ import annotations

from hearts.api.game_store import store
from hearts.api.serialize import card_to_id, sort_cards
from hearts.cards import Card
from hearts.rules import NUM_PLAYERS
from hearts.state import GameState, Phase


def _ids(cards: list[Card]) -> list[str]:
    return [card_to_id(card) for card in sort_cards(cards)]


def _tricks_won_counts(state: GameState) -> list[int]:
    return [len(cards) // NUM_PLAYERS for cards in state.tricks_won]


def _trick_payload(trick: list[tuple[int, Card]]) -> list[dict]:
    return [{"player": player, "card": card_to_id(card)} for player, card in trick]


def opening_frame(state: GameState) -> dict:
    """Table at the first lead, before any card of this round."""
    return {
        "kind": "opening",
        "actor": None,
        "card": None,
        "legal_moves": [],
        "decision_hand": None,
        "hands": [_ids(state.hands[seat]) for seat in range(NUM_PLAYERS)],
        "current_trick": [],
        "tricks_won_counts": _tricks_won_counts(state),
        "hearts_broken": state.hearts_broken,
        "trick_number": state.trick_number,
        "current_player": state.current_player,
        "trick_winner": None,
    }


def play_frame(before: GameState, after: GameState, player: int, card: Card) -> dict:
    """The table as ``card`` lands, with the actor's pre-play hand kept for review.

    ``hands`` is after the card leaves the hand, matching the card on the trick.
    ``decision_hand`` is what that seat held while choosing, including ``card``.
    A completed trick stays on the table; it is not in the won-trick counts yet.
    """
    trick = list(before.current_trick) + [(player, card)]
    hands = []
    for seat in range(NUM_PLAYERS):
        seat_cards = list(before.hands[seat])
        if seat == player:
            seat_cards.remove(card)
        hands.append(_ids(seat_cards))
    legal = _ids(before.legal_moves_for(player))
    complete = len(trick) == NUM_PLAYERS
    return {
        "kind": "play",
        "actor": player,
        "card": card_to_id(card),
        "legal_moves": legal,
        "decision_hand": _ids(before.hands[player]),
        "hands": hands,
        "current_trick": _trick_payload(trick),
        "tricks_won_counts": _tricks_won_counts(before),
        "hearts_broken": after.hearts_broken,
        "trick_number": before.trick_number,
        "current_player": player,
        "trick_winner": after.trick_leader if complete else None,
    }


def reset(game_id: str) -> None:
    store.reset_replay(game_id)


def note_opening(game_id: str, state: GameState) -> None:
    if state.phase is not Phase.PLAYING:
        return
    if game_id in store.openings:
        return
    store.openings[game_id] = opening_frame(state)


def note_play(game_id: str, before: GameState, after: GameState, player: int, card: Card) -> None:
    store.plays.setdefault(game_id, []).append(play_frame(before, after, player, card))


def payload(game_id: str) -> dict:
    return {
        "opening": store.openings.get(game_id),
        "plays": list(store.plays.get(game_id, [])),
    }
