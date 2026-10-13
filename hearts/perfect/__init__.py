"""Perfect-information Hearts: encoder, policy/value net, paranoid MCTS, self-play loop.

Everything in this package sees all four hands. It depends only on the rules
engine (``hearts.cards``, ``hearts.rules``, ``hearts.state``, ``hearts.config``).
"""

from hearts.perfect.encode import (
    GLOBAL_DIM,
    NUM_LOCATIONS,
    PerfectObs,
    encode_perfect,
    legal_mask_for,
    normalize_points,
    rotate_to_absolute,
    rotate_to_relative,
)
from hearts.perfect.game import card_from_id, new_hold_round
from hearts.perfect.mcts import MCTSConfig, MCTSResult, run_mcts, select_action
from hearts.perfect.model import PerfectNet, PerfectOutput

__all__ = [
    "GLOBAL_DIM",
    "MCTSConfig",
    "MCTSResult",
    "NUM_LOCATIONS",
    "PerfectNet",
    "PerfectObs",
    "PerfectOutput",
    "card_from_id",
    "encode_perfect",
    "legal_mask_for",
    "new_hold_round",
    "normalize_points",
    "rotate_to_absolute",
    "rotate_to_relative",
    "run_mcts",
    "select_action",
]
