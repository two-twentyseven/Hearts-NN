from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hearts.cards import Card
from hearts.rules import CARDS_PER_HAND, NUM_PLAYERS, score_captured_cards
from hearts.state import DECK_SIZE, GameState

POINTS_PER_ROUND = 26.0
NONE_INDEX = -1


@dataclass(frozen=True)
class EncodedObservation:
    """Egocentric observation for the acting player during play.

    Card arrays are indexed by ``Card.card_id`` (suit_index * 13 + rank_index).
    Seat indices are relative: 0 is the acting player, then clockwise.
    Unplayed / unknown categoricals use ``NONE_INDEX`` (-1).
    """

    in_hand: np.ndarray  # (52,) float32 {0,1}
    played: np.ndarray  # (52,) float32 {0,1}
    played_by: np.ndarray  # (52,) int64 in {-1,0,1,2,3}
    played_on_trick: np.ndarray  # (52,) int64 in {-1,0..12}
    play_position: np.ndarray  # (52,) int64 in {-1,0..3}; 0=leader … 3=closer
    broke_suit: np.ndarray  # (52,) float32 {0,1}; 1 if played off the led suit
    trick_leaders: np.ndarray  # (13,) int64 in {-1,0..3}
    trick_led_suits: np.ndarray  # (13,) int64 in {-1,0..3}; led suit per trick
    suit_led: int  # {-1,0..3}
    my_role: int  # {0..3}
    hearts_broken: float  # {0,1}
    points_taken: np.ndarray  # (4,) float32
    current_trick: int  # {0..12}


def relative_seat(absolute_seat: int, acting_player: int) -> int:
    """Map an engine seat onto an egocentric seat (0 = acting player, clockwise)."""
    return (absolute_seat - acting_player) % NUM_PLAYERS


def encode_state(game_state: GameState, acting_player: int) -> EncodedObservation:
    """Encode `game_state` from `acting_player`'s view as structured card + global features.

    Passing-phase extras and match (cross-round) scores are not encoded.
    """
    in_hand = np.zeros(DECK_SIZE, dtype=np.float32)
    for card in game_state.hands[acting_player]:
        in_hand[card.card_id] = 1.0

    played = np.zeros(DECK_SIZE, dtype=np.float32)
    played_by = np.full(DECK_SIZE, NONE_INDEX, dtype=np.int64)
    played_on_trick = np.full(DECK_SIZE, NONE_INDEX, dtype=np.int64)
    for absolute_seat, by_card in enumerate(game_state.card_play_tricks):
        rel = relative_seat(absolute_seat, acting_player)
        for card_id, trick_frac in enumerate(by_card):
            if trick_frac <= 0.0:
                continue
            # Engine stores trick_number / 13; recover 0-based trick index.
            trick_index = int(round(trick_frac * CARDS_PER_HAND)) - 1
            played[card_id] = 1.0
            played_by[card_id] = rel
            played_on_trick[card_id] = trick_index

    trick_leaders = _encode_trick_leaders(game_state, acting_player)
    trick_led_suits = _encode_trick_led_suits(trick_leaders, played, played_by, played_on_trick)
    play_position, broke_suit = _encode_play_context(
        played, played_by, played_on_trick, trick_leaders, trick_led_suits
    )
    suit_led = _encode_suit_led(game_state)
    my_role = _encode_my_role(game_state, acting_player)
    points_taken = _encode_points_taken(game_state, acting_player)
    trick_number = min(max(game_state.trick_number, 1), CARDS_PER_HAND)
    current_trick = trick_number - 1

    return EncodedObservation(
        in_hand=in_hand,
        played=played,
        played_by=played_by,
        played_on_trick=played_on_trick,
        play_position=play_position,
        broke_suit=broke_suit,
        trick_leaders=trick_leaders,
        trick_led_suits=trick_led_suits,
        suit_led=suit_led,
        my_role=my_role,
        hearts_broken=1.0 if game_state.hearts_broken else 0.0,
        points_taken=points_taken,
        current_trick=current_trick,
    )


def _encode_trick_leaders(game_state: GameState, acting_player: int) -> np.ndarray:
    leaders = np.full(CARDS_PER_HAND, NONE_INDEX, dtype=np.int64)
    current_index = min(game_state.trick_number, CARDS_PER_HAND) - 1
    for trick_index in range(current_index + 1):
        leader = game_state.trick_leaders[trick_index]
        if leader is None:
            if trick_index == game_state.trick_number - 1:
                leader = game_state.trick_leader
            else:
                continue
        leaders[trick_index] = relative_seat(leader, acting_player)
    return leaders


def _suit_index_from_card_id(card_id: int) -> int:
    return card_id // 13


def _encode_trick_led_suits(
    trick_leaders: np.ndarray,
    played: np.ndarray,
    played_by: np.ndarray,
    played_on_trick: np.ndarray,
) -> np.ndarray:
    led = np.full(CARDS_PER_HAND, NONE_INDEX, dtype=np.int64)
    for trick_index in range(CARDS_PER_HAND):
        leader = trick_leaders[trick_index]
        if leader == NONE_INDEX:
            continue
        for card_id in range(DECK_SIZE):
            if (
                played[card_id] == 1.0
                and played_on_trick[card_id] == trick_index
                and played_by[card_id] == leader
            ):
                led[trick_index] = _suit_index_from_card_id(card_id)
                break
    return led


def _encode_play_context(
    played: np.ndarray,
    played_by: np.ndarray,
    played_on_trick: np.ndarray,
    trick_leaders: np.ndarray,
    trick_led_suits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    play_position = np.full(DECK_SIZE, NONE_INDEX, dtype=np.int64)
    broke_suit = np.zeros(DECK_SIZE, dtype=np.float32)
    for card_id in range(DECK_SIZE):
        if played[card_id] != 1.0:
            continue
        trick_index = int(played_on_trick[card_id])
        leader = int(trick_leaders[trick_index])
        seat = int(played_by[card_id])
        play_position[card_id] = (seat - leader) % NUM_PLAYERS
        led_suit = int(trick_led_suits[trick_index])
        if led_suit != NONE_INDEX:
            broke_suit[card_id] = (
                1.0 if _suit_index_from_card_id(card_id) != led_suit else 0.0
            )
    return play_position, broke_suit


def _encode_suit_led(game_state: GameState) -> int:
    if not game_state.current_trick:
        return NONE_INDEX
    return game_state.current_trick[0][1].suit_index


def _encode_my_role(game_state: GameState, acting_player: int) -> int:
    current_index = min(max(game_state.trick_number, 1), CARDS_PER_HAND) - 1
    leader = game_state.trick_leaders[current_index]
    if leader is None:
        leader = game_state.trick_leader
    leader_relative = relative_seat(leader, acting_player)
    return (NUM_PLAYERS - leader_relative) % NUM_PLAYERS


def _encode_points_taken(game_state: GameState, acting_player: int) -> np.ndarray:
    points = np.zeros(NUM_PLAYERS, dtype=np.float32)
    for absolute_seat, cards in enumerate(game_state.tricks_won):
        rel = relative_seat(absolute_seat, acting_player)
        points[rel] = score_captured_cards(cards, game_state.config) / POINTS_PER_ROUND
    return points
