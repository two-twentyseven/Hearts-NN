from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from hearts.perfect.encode import encode_perfect, legal_mask_for
from hearts.perfect.game import card_from_id, is_round_over, new_hold_round
from hearts.perfect.mcts import MCTSConfig, run_mcts
from hearts.perfect.model import PerfectNet, load_checkpoint, predict
from hearts.rules import NUM_PLAYERS
from hearts.state import GameState


class Agent(Protocol):
    name: str

    def choose(self, state: GameState, rng: np.random.Generator) -> int:
        """Return the card id to play for ``state.current_player``."""


class RandomAgent:
    name = "random"

    def choose(self, state: GameState, rng: np.random.Generator) -> int:
        legal = state.legal_moves_for(state.current_player)
        return int(legal[rng.integers(len(legal))].card_id)


class PolicyAgent:
    """Greedy raw policy head, no search."""

    name = "policy"

    def __init__(self, net: PerfectNet) -> None:
        self.net = net.eval()

    @torch.inference_mode()
    def choose(self, state: GameState, rng: np.random.Generator) -> int:
        acting = state.current_player
        priors, _ = predict(self.net, [encode_perfect(state, acting)], [legal_mask_for(state, acting)])
        return int(np.argmax(priors[0]))


class MCTSAgent:
    def __init__(self, net: PerfectNet, cfg: MCTSConfig, name: str = "mcts") -> None:
        self.net = net.eval()
        self.cfg = cfg
        self.name = name

    def choose(self, state: GameState, rng: np.random.Generator) -> int:
        result = run_mcts(state, self.net, self.cfg, rng, add_noise=False)
        return int(np.argmax(result.visit_counts))


@dataclass
class ArenaResult:
    n_games: int
    mean_points_a: float
    mean_points_b: float  # averaged over the three B seats
    per_game: list[list[int]]


def play_match_round(
    agents: list[Agent],
    *,
    seed: int | None,
    rng: np.random.Generator,
) -> list[int]:
    state = new_hold_round(seed=seed)
    while not is_round_over(state):
        acting = state.current_player
        card_id = agents[acting].choose(state, rng)
        state = state.play_card(acting, card_from_id(card_id))
    return list(state.last_round_scores)


def evaluate(
    agent_a: Agent,
    agent_b: Agent,
    n_games: int,
    *,
    base_seed: int = 0,
    progress: bool = False,
) -> ArenaResult:
    """Agent A takes one seat (rotating each game); B fills the other three."""
    rng = np.random.default_rng(base_seed)
    a_points: list[float] = []
    b_points: list[float] = []
    per_game: list[list[int]] = []
    for i in range(n_games):
        seat_a = i % NUM_PLAYERS
        agents: list[Agent] = [agent_b] * NUM_PLAYERS
        agents[seat_a] = agent_a
        scores = play_match_round(agents, seed=base_seed + i, rng=rng)
        per_game.append(scores)
        a_points.append(scores[seat_a])
        b_points.append(sum(s for p, s in enumerate(scores) if p != seat_a) / (NUM_PLAYERS - 1))
        if progress:
            print(f"  arena game {i + 1}/{n_games}: A(seat {seat_a})={scores[seat_a]} scores={scores}", flush=True)
    return ArenaResult(
        n_games=n_games,
        mean_points_a=float(np.mean(a_points)),
        mean_points_b=float(np.mean(b_points)),
        per_game=per_game,
    )


# ---------------------------------------------------------------------------
# Declarative agents + parallel head-to-head with statistics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentSpec:
    """Picklable description of an agent so worker processes can build their own."""

    kind: str  # "random" | "policy" | "mcts"
    checkpoint: str | None = None
    sims: int = 200
    opponent_mode: str = "paranoid"

    @property
    def label(self) -> str:
        if self.kind == "random":
            return "random"
        name = Path(self.checkpoint).name if self.checkpoint else "?"
        if self.kind == "policy":
            return f"policy[{name}]"
        return f"mcts{self.sims}[{name}]"


def build_agent(spec: AgentSpec, cache: dict[str, PerfectNet] | None = None) -> Agent:
    if spec.kind == "random":
        return RandomAgent()
    if spec.checkpoint is None:
        raise ValueError(f"{spec.kind} agent needs a checkpoint")
    cache = cache if cache is not None else {}
    net = cache.get(spec.checkpoint)
    if net is None:
        net, _ = load_checkpoint(spec.checkpoint)
        cache[spec.checkpoint] = net
    if spec.kind == "policy":
        return PolicyAgent(net)
    if spec.kind == "mcts":
        return MCTSAgent(
            net,
            MCTSConfig(n_simulations=spec.sims, dirichlet_eps=0.0, opponent_mode=spec.opponent_mode),
            name=spec.label,
        )
    raise ValueError(f"Unknown agent kind {spec.kind!r}")


@dataclass
class GameOutcome:
    seed: int
    seats_a: tuple[int, ...]
    scores: list[int]


@dataclass
class MatchStats:
    n_games: int
    seats_a_per_game: int
    mean_a: float
    mean_b: float
    diff: float  # mean_b - mean_a; positive means A takes fewer points
    diff_ci95: float  # half-width of the 95% CI on the per-game A-vs-B gap
    a_lowest_rate: float  # fraction of games where an A seat had the (unique or tied) lowest score
    a_below_b_rate: float  # fraction of games where mean A points < mean B points
    a_moons: int
    b_moons: int
    outcomes: list[GameOutcome]


def _seats_for_game(game_index: int, seats_a_per_game: int) -> tuple[int, ...]:
    start = game_index % NUM_PLAYERS
    return tuple((start + k) % NUM_PLAYERS for k in range(seats_a_per_game))


def _play_one(agent_a: Agent, agent_b: Agent, seed: int, seats_a: tuple[int, ...]) -> GameOutcome:
    agents: list[Agent] = [agent_b] * NUM_PLAYERS
    for s in seats_a:
        agents[s] = agent_a
    rng = np.random.default_rng(seed)
    scores = play_match_round(agents, seed=seed, rng=rng)
    return GameOutcome(seed=seed, seats_a=seats_a, scores=scores)


_W_A: Agent | None = None
_W_B: Agent | None = None


def _match_worker_init(spec_a: dict, spec_b: dict) -> None:
    global _W_A, _W_B
    torch.set_num_threads(1)
    cache: dict[str, PerfectNet] = {}
    _W_A = build_agent(AgentSpec(**spec_a), cache)
    _W_B = build_agent(AgentSpec(**spec_b), cache)


def _match_worker_play(job: tuple[int, tuple[int, ...]]) -> GameOutcome:
    assert _W_A is not None and _W_B is not None
    seed, seats_a = job
    return _play_one(_W_A, _W_B, seed, seats_a)


def summarize(outcomes: list[GameOutcome], seats_a_per_game: int) -> MatchStats:
    a_means, b_means, lowest, moons_a, moons_b = [], [], [], 0, 0
    for o in outcomes:
        a = [o.scores[s] for s in o.seats_a]
        b = [o.scores[s] for s in range(NUM_PLAYERS) if s not in o.seats_a]
        a_means.append(float(np.mean(a)))
        b_means.append(float(np.mean(b)))
        lowest.append(min(a) <= min(b))
        if sum(o.scores) == 78:  # shoot the moon: shooter has 0, others 26
            shooter = o.scores.index(0)
            if shooter in o.seats_a:
                moons_a += 1
            else:
                moons_b += 1
    gaps = np.asarray(b_means) - np.asarray(a_means)
    n = len(outcomes)
    ci = 1.96 * float(gaps.std(ddof=1)) / math.sqrt(n) if n > 1 else float("nan")
    return MatchStats(
        n_games=n,
        seats_a_per_game=seats_a_per_game,
        mean_a=float(np.mean(a_means)),
        mean_b=float(np.mean(b_means)),
        diff=float(gaps.mean()),
        diff_ci95=ci,
        a_lowest_rate=float(np.mean(lowest)),
        a_below_b_rate=float(np.mean(gaps > 0)),
        a_moons=moons_a,
        b_moons=moons_b,
        outcomes=outcomes,
    )


def run_match(
    spec_a: AgentSpec,
    spec_b: AgentSpec,
    n_games: int,
    *,
    seats_a_per_game: int = 1,
    base_seed: int = 0,
    workers: int = 1,
    progress: bool = False,
) -> MatchStats:
    """Play ``n_games`` hold rounds of A vs B.

    ``seats_a_per_game`` = 1 puts one A among three B (rotating seat each game);
    3 puts three A against one B (what a human faces in the GUI).
    """
    if not 1 <= seats_a_per_game <= NUM_PLAYERS - 1:
        raise ValueError("seats_a_per_game must be 1..3")
    jobs = [(base_seed + i, _seats_for_game(i, seats_a_per_game)) for i in range(n_games)]
    outcomes: list[GameOutcome] = []
    if workers <= 1:
        cache: dict[str, PerfectNet] = {}
        agent_a = build_agent(spec_a, cache)
        agent_b = build_agent(spec_b, cache)
        for i, (seed, seats_a) in enumerate(jobs):
            outcomes.append(_play_one(agent_a, agent_b, seed, seats_a))
            if progress:
                print(f"  game {i + 1}/{n_games}: {outcomes[-1].scores} A seats {seats_a}", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=min(workers, n_games),
            initializer=_match_worker_init,
            initargs=(asdict(spec_a), asdict(spec_b)),
        ) as pool:
            for i, outcome in enumerate(pool.imap_unordered(_match_worker_play, jobs)):
                outcomes.append(outcome)
                if progress and (i + 1) % max(1, n_games // 20) == 0:
                    print(f"  {i + 1}/{n_games} games done", flush=True)
    outcomes.sort(key=lambda o: o.seed)
    return summarize(outcomes, seats_a_per_game)
