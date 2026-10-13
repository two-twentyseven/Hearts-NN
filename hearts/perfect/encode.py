from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hearts.rules import CARDS_PER_HAND, NUM_PLAYERS, score_captured_cards
from hearts.state import DECK_SIZE, GameState

POINTS_PER_ROUND = 26.0

# Location kinds for each card. Location index = kind * 4 + relative_seat.
LOC_IN_HAND = 0
LOC_ON_TRICK = 1
LOC_TAKEN = 2
NUM_LOCATION_KINDS = 3
NUM_LOCATIONS = NUM_LOCATION_KINDS * NUM_PLAYERS  # 12

# Global feature layout.
_G_ROUND_POINTS = 0  # 4: points taken this round, / 26
_G_GAME_POINTS = 4  # 4: cumulative match points, / 100
_G_LEADER = 8  # 4: one-hot relative seat of the current trick leader
_G_MY_POSITION = 12  # 4: one-hot position of the acting player in the current trick
_G_HEARTS_BROKEN = 16  # 1
_G_TRICK_NUMBER = 17  # 1: (trick_number - 1) / 12
GLOBAL_DIM = 18


@dataclass(frozen=True)
class PerfectObs:
    """Egocentric perfect-information observation.

    ``locations[card_id]`` is in ``0..11``: ``kind * 4 + rel_seat`` where kind is
    in-hand (0), on the current trick (1) or already taken in a won trick (2),
    and rel_seat is 0 for the acting player, 1 for the seat to their left,
    2 across, 3 to their right.
    """

    locations: np.ndarray  # (52,) int64
    globals: np.ndarray  # (GLOBAL_DIM,) float32


def relative_seat(absolute_seat: int, acting: int) -> int:
    return (absolute_seat - acting) % NUM_PLAYERS


def absolute_seat(relative: int, acting: int) -> int:
    return (relative + acting) % NUM_PLAYERS


def rotate_to_relative(values_abs, acting: int) -> np.ndarray:
    """Reorder a length-4 per-seat vector from absolute seats to the acting player's view.

    ``result[rel] = values_abs[absolute_seat(rel, acting)]``.
    """
    arr = np.asarray(values_abs)
    return arr[[absolute_seat(rel, acting) for rel in range(NUM_PLAYERS)]]


def rotate_to_absolute(values_rel, acting: int) -> np.ndarray:
    """Inverse of :func:`rotate_to_relative`: ``result[abs] = values_rel[relative_seat(abs, acting)]``."""
    arr = np.asarray(values_rel)
    return arr[[relative_seat(seat, acting) for seat in range(NUM_PLAYERS)]]


def normalize_points(points: float) -> float:
    """Map round points to z in [-1, 1]; fewer points is better (higher z)."""
    z = 1.0 - 2.0 * (points / POINTS_PER_ROUND)
    return float(max(-1.0, min(1.0, z)))


def normalize_scores(round_scores) -> np.ndarray:
    return np.asarray([normalize_points(p) for p in round_scores], dtype=np.float32)


def encode_perfect(state: GameState, acting: int) -> PerfectObs:
    """Encode ``state`` from ``acting``'s seat with every card's holder visible."""
    locations = np.full(DECK_SIZE, -1, dtype=np.int64)
    for seat in range(NUM_PLAYERS):
        rel = relative_seat(seat, acting)
        for card in state.hands[seat]:
            locations[card.card_id] = LOC_IN_HAND * NUM_PLAYERS + rel
        for card in state.tricks_won[seat]:
            locations[card.card_id] = LOC_TAKEN * NUM_PLAYERS + rel
    for seat, card in state.current_trick:
        locations[card.card_id] = LOC_ON_TRICK * NUM_PLAYERS + relative_seat(seat, acting)
    if (locations < 0).any():
        missing = int((locations < 0).sum())
        raise ValueError(f"{missing} cards have no location; state is not a full deal")

    g = np.zeros(GLOBAL_DIM, dtype=np.float32)
    for seat in range(NUM_PLAYERS):
        rel = relative_seat(seat, acting)
        g[_G_ROUND_POINTS + rel] = score_captured_cards(state.tricks_won[seat], state.config) / POINTS_PER_ROUND
        g[_G_GAME_POINTS + rel] = state.scores[seat] / 100.0 if state.scores else 0.0
    leader_rel = relative_seat(state.trick_leader, acting)
    g[_G_LEADER + leader_rel] = 1.0
    g[_G_MY_POSITION + (NUM_PLAYERS - leader_rel) % NUM_PLAYERS] = 1.0
    g[_G_HEARTS_BROKEN] = 1.0 if state.hearts_broken else 0.0
    trick_number = min(max(state.trick_number, 1), CARDS_PER_HAND)
    g[_G_TRICK_NUMBER] = (trick_number - 1) / float(CARDS_PER_HAND - 1)
    return PerfectObs(locations=locations, globals=g)


def legal_mask_for(state: GameState, player: int | None = None) -> np.ndarray:
    """Length-52 bool mask of legal plays for ``player`` (default: to-move)."""
    player = state.current_player if player is None else player
    mask = np.zeros(DECK_SIZE, dtype=bool)
    for card in state.legal_moves_for(player):
        mask[card.card_id] = True
    return mask
