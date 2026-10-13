from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from hearts.perfect.encode import (
    encode_perfect,
    legal_mask_for,
    normalize_scores,
    rotate_to_absolute,
)
from hearts.perfect.game import card_from_id, is_round_over, strip_history
from hearts.perfect.model import PerfectNet, predict
from hearts.rules import NUM_PLAYERS
from hearts.state import DECK_SIZE, GameState

OPPONENT_MODES = ("paranoid", "self_interest")


@dataclass(frozen=True)
class MCTSConfig:
    n_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.6
    dirichlet_eps: float = 0.25
    # "paranoid": opponents pick the move that minimizes the root player's value.
    # "self_interest": every seat maximizes its own predicted value.
    opponent_mode: str = "paranoid"
    # Unvisited children start from the parent's mean value minus this amount
    # (from the selecting seat's perspective), so they are neither free nor forbidden.
    fpu_reduction: float = 0.0
    skip_single_move: bool = True
    # Inside the tree, play through positions with exactly one legal card without
    # spending a simulation on them (they contain no decision).
    collapse_forced_moves: bool = True

    def __post_init__(self) -> None:
        if self.opponent_mode not in OPPONENT_MODES:
            raise ValueError(f"opponent_mode must be one of {OPPONENT_MODES}")


@dataclass
class MCTSResult:
    visit_counts: np.ndarray  # (52,) int
    pi: np.ndarray  # (52,) float, visit distribution over legal cards
    root_value: np.ndarray  # (4,) mean backed-up value, absolute seat order
    q_values: np.ndarray  # (52,) root player's mean value after each visited action, NaN otherwise
    action_ids: list[int] = field(default_factory=list)  # legal card ids at the root
    root: "Node | None" = None  # search tree, for inspection/debugging


class Node:
    __slots__ = ("state", "to_move", "legal_ids", "priors", "N", "W", "children", "terminal_value")

    def __init__(self, state: GameState) -> None:
        self.state = state
        self.to_move = state.current_player
        self.legal_ids: list[int] = []
        self.priors: np.ndarray | None = None  # (52,) once expanded
        self.N = 0
        self.W = np.zeros(NUM_PLAYERS, dtype=np.float64)
        self.children: dict[int, Node] = {}
        self.terminal_value: np.ndarray | None = None
        if is_round_over(state):
            self.terminal_value = normalize_scores(state.last_round_scores).astype(np.float64)

    @property
    def is_terminal(self) -> bool:
        return self.terminal_value is not None

    @property
    def expanded(self) -> bool:
        return self.priors is not None

    def mean_value(self) -> np.ndarray:
        if self.N == 0:
            return np.zeros(NUM_PLAYERS)
        return self.W / self.N


def _perspective_and_sign(node: Node, root_player: int, cfg: MCTSConfig) -> tuple[int, float]:
    """Which seat's value the selecting node cares about, and whether it maximizes (+1) or minimizes (-1)."""
    if cfg.opponent_mode == "self_interest":
        return node.to_move, 1.0
    if node.to_move == root_player:
        return root_player, 1.0
    return root_player, -1.0


def _select_child(node: Node, root_player: int, cfg: MCTSConfig) -> int:
    persp, sign = _perspective_and_sign(node, root_player, cfg)
    parent_q = sign * node.mean_value()[persp] - cfg.fpu_reduction
    sqrt_n = math.sqrt(max(node.N, 1))
    best_score = -math.inf
    best_action = node.legal_ids[0]
    priors = node.priors
    for a in node.legal_ids:
        child = node.children.get(a)
        if child is None or child.N == 0:
            q = parent_q
            n = 0
        else:
            q = sign * (child.W[persp] / child.N)
            n = child.N
        u = cfg.c_puct * priors[a] * sqrt_n / (1 + n)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = a
    return best_action


def _expand(node: Node, net: PerfectNet) -> np.ndarray:
    """Run the net on a non-terminal leaf. Returns the value in absolute seat order."""
    obs = encode_perfect(node.state, node.to_move)
    mask = legal_mask_for(node.state, node.to_move)
    priors, values = predict(net, [obs], [mask])
    node.priors = priors[0].astype(np.float64)
    node.legal_ids = [int(i) for i in np.flatnonzero(mask)]
    return rotate_to_absolute(values[0], node.to_move).astype(np.float64)


def _fast_forward_forced(state: GameState) -> GameState:
    """Advance through consecutive positions that have exactly one legal card."""
    while not is_round_over(state):
        legal = state.legal_moves_for(state.current_player)
        if len(legal) != 1:
            break
        state = state.play_card(state.current_player, legal[0])
    return state


def _make_child(state: GameState, cfg: MCTSConfig) -> Node:
    if cfg.collapse_forced_moves:
        state = _fast_forward_forced(state)
    return Node(state)


def _add_dirichlet_noise(node: Node, cfg: MCTSConfig, rng: np.random.Generator) -> None:
    if cfg.dirichlet_eps <= 0 or len(node.legal_ids) < 2:
        return
    noise = rng.dirichlet([cfg.dirichlet_alpha] * len(node.legal_ids))
    for a, eta in zip(node.legal_ids, noise):
        node.priors[a] = (1 - cfg.dirichlet_eps) * node.priors[a] + cfg.dirichlet_eps * eta


def run_mcts(
    state: GameState,
    net: PerfectNet,
    cfg: MCTSConfig,
    rng: np.random.Generator | None = None,
    *,
    add_noise: bool = True,
) -> MCTSResult:
    """Paranoid PUCT from the to-move player's seat on a fully known deal.

    The root player's nodes maximize the root player's predicted post-round
    value; opponent nodes minimize it (``cfg.opponent_mode == "paranoid"``).
    Leaves are scored by the value head (no rollouts); terminal nodes use the
    true post-round scores.
    """
    if is_round_over(state):
        raise ValueError("Cannot search a finished round")
    rng = rng if rng is not None else np.random.default_rng()
    root_player = state.current_player
    root = Node(strip_history(state))
    root_value = _expand(root, net)
    root.N += 1
    root.W += root_value

    if cfg.skip_single_move and len(root.legal_ids) == 1:
        return _single_move_result(root, root_player)

    if add_noise:
        _add_dirichlet_noise(root, cfg, rng)

    for _ in range(cfg.n_simulations):
        node = root
        path = [root]
        while node.expanded and not node.is_terminal:
            a = _select_child(node, root_player, cfg)
            child = node.children.get(a)
            if child is None:
                next_state = node.state.play_card(node.to_move, card_from_id(a))
                child = _make_child(next_state, cfg)
                node.children[a] = child
            node = child
            path.append(node)
        if node.is_terminal:
            value = node.terminal_value
        else:
            value = _expand(node, net)
        for n in path:
            n.N += 1
            n.W += value

    visit_counts = np.zeros(DECK_SIZE, dtype=np.int64)
    q_values = np.full(DECK_SIZE, np.nan, dtype=np.float64)
    for a, child in root.children.items():
        visit_counts[a] = child.N
        if child.N > 0:
            q_values[a] = child.W[root_player] / child.N
    total = visit_counts.sum()
    pi = visit_counts / total if total > 0 else np.zeros(DECK_SIZE)
    return MCTSResult(
        visit_counts=visit_counts,
        pi=pi.astype(np.float32),
        root_value=root.mean_value(),
        q_values=q_values,
        action_ids=list(root.legal_ids),
        root=root,
    )


def _single_move_result(root: Node, root_player: int) -> MCTSResult:
    a = root.legal_ids[0]
    visit_counts = np.zeros(DECK_SIZE, dtype=np.int64)
    visit_counts[a] = 1
    pi = np.zeros(DECK_SIZE, dtype=np.float32)
    pi[a] = 1.0
    q_values = np.full(DECK_SIZE, np.nan)
    q_values[a] = root.mean_value()[root_player]
    return MCTSResult(
        visit_counts=visit_counts,
        pi=pi,
        root_value=root.mean_value(),
        q_values=q_values,
        action_ids=[a],
        root=root,
    )


def select_action(pi: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """Pick a card id from a visit distribution. ``temperature <= 0`` is greedy."""
    if temperature <= 0:
        return int(np.argmax(pi))
    logits = np.where(pi > 0, np.log(np.maximum(pi, 1e-12)) / temperature, -np.inf)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(rng.choice(DECK_SIZE, p=probs))
