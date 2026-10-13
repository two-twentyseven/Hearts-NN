from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch

from hearts.config import GameConfig
from hearts.perfect.corpus import concat_shards, make_shard
from hearts.perfect.encode import (
    encode_perfect,
    legal_mask_for,
    normalize_scores,
    rotate_to_relative,
)
from hearts.perfect.game import card_from_id, is_round_over, new_hold_round
from hearts.perfect.mcts import MCTSConfig, run_mcts, select_action
from hearts.perfect.model import NetConfig, PerfectNet


@dataclass(frozen=True)
class SelfPlayConfig:
    temperature: float = 1.0  # sampling temperature over visit counts for early tricks
    temperature_cutoff_trick: int = 7  # from this trick (1-based) on, play greedily
    final_temperature: float = 0.0


@dataclass
class RoundData:
    shard: dict
    round_scores: list[int]
    seed: int | None
    actors: list[int]  # absolute seat that made each recorded decision (all four seats contribute)


@torch.inference_mode()
def play_round(
    net: PerfectNet,
    mcts_cfg: MCTSConfig,
    play_cfg: SelfPlayConfig | None = None,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    config: GameConfig | None = None,
) -> RoundData:
    """Play one hold round where every seat runs paranoid MCTS rooted at itself.

    Returns one corpus row per decision, for all four seats (52 rows per round):
    the actor's perfect-info encoding, the legal mask, the MCTS visit
    distribution as policy target, and the true normalized post-round scores
    rotated to the actor's seat order as value target.
    """
    play_cfg = play_cfg or SelfPlayConfig()
    rng = rng if rng is not None else np.random.default_rng(seed)
    net.eval()
    state = new_hold_round(seed=seed, config=config)

    locations: list[np.ndarray] = []
    globals_: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    pis: list[np.ndarray] = []
    actors: list[int] = []

    while not is_round_over(state):
        acting = state.current_player
        obs = encode_perfect(state, acting)
        mask = legal_mask_for(state, acting)
        result = run_mcts(state, net, mcts_cfg, rng, add_noise=True)
        temperature = (
            play_cfg.temperature
            if state.trick_number < play_cfg.temperature_cutoff_trick
            else play_cfg.final_temperature
        )
        card_id = select_action(result.pi, temperature, rng)

        locations.append(obs.locations)
        globals_.append(obs.globals)
        masks.append(mask)
        pis.append(result.pi)
        actors.append(acting)
        state = state.play_card(acting, card_from_id(card_id))

    round_scores = list(state.last_round_scores)
    z_abs = normalize_scores(round_scores)
    z = np.stack([rotate_to_relative(z_abs, a) for a in actors]).astype(np.float32)
    shard = make_shard(
        np.stack(locations),
        np.stack(globals_),
        np.stack(masks),
        np.stack(pis),
        z,
        meta={"seed": seed, "round_scores": round_scores, "n_simulations": mcts_cfg.n_simulations},
    )
    return RoundData(shard=shard, round_scores=round_scores, seed=seed, actors=actors)


# ---------------------------------------------------------------------------
# Multi-process generation
# ---------------------------------------------------------------------------

_WORKER_NET: PerfectNet | None = None
_WORKER_MCTS: MCTSConfig | None = None
_WORKER_PLAY: SelfPlayConfig | None = None


def _shard_to_numpy(shard: dict) -> dict:
    """Tensors -> numpy so shards can cross process boundaries without torch shared memory."""
    return {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in shard.items()}


def _shard_from_numpy(shard: dict) -> dict:
    return {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for k, v in shard.items()}


def _worker_init(state_dict_np: dict, net_config: dict, mcts_cfg: dict, play_cfg: dict) -> None:
    global _WORKER_NET, _WORKER_MCTS, _WORKER_PLAY
    torch.set_num_threads(1)
    net = PerfectNet(NetConfig(**net_config))
    net.load_state_dict({k: torch.from_numpy(v) for k, v in state_dict_np.items()})
    net.eval()
    _WORKER_NET = net
    _WORKER_MCTS = MCTSConfig(**mcts_cfg)
    _WORKER_PLAY = SelfPlayConfig(**play_cfg)


def _worker_play(seed: int) -> tuple[dict, list[int]]:
    assert _WORKER_NET is not None and _WORKER_MCTS is not None and _WORKER_PLAY is not None
    data = play_round(_WORKER_NET, _WORKER_MCTS, _WORKER_PLAY, seed=seed)
    return _shard_to_numpy(data.shard), data.round_scores


def generate_games(
    net: PerfectNet,
    n_games: int,
    *,
    base_seed: int,
    mcts_cfg: MCTSConfig,
    play_cfg: SelfPlayConfig | None = None,
    workers: int = 1,
    progress: bool = False,
) -> tuple[dict, list[list[int]]]:
    """Play ``n_games`` hold rounds and return one merged shard plus per-game round scores."""
    play_cfg = play_cfg or SelfPlayConfig()
    seeds = [base_seed + i for i in range(n_games)]
    shards: list[dict] = []
    scores: list[list[int]] = []

    if workers <= 1:
        for i, seed in enumerate(seeds):
            data = play_round(net, mcts_cfg, play_cfg, seed=seed)
            shards.append(data.shard)
            scores.append(data.round_scores)
            if progress:
                print(f"  game {i + 1}/{n_games} scores={data.round_scores}", flush=True)
    else:
        state_dict_np = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=min(workers, n_games),
            initializer=_worker_init,
            initargs=(state_dict_np, net.config.to_dict(), asdict(mcts_cfg), asdict(play_cfg)),
        ) as pool:
            for i, (shard_np, round_scores) in enumerate(pool.imap_unordered(_worker_play, seeds)):
                shards.append(_shard_from_numpy(shard_np))
                scores.append(round_scores)
                if progress:
                    print(f"  game {i + 1}/{n_games} scores={round_scores}", flush=True)

    merged = concat_shards(shards)
    merged["meta"] = {
        "base_seed": base_seed,
        "n_games": n_games,
        "n_simulations": mcts_cfg.n_simulations,
        "round_scores": scores,
    }
    return merged, scores


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


__all__ = [
    "RoundData",
    "SelfPlayConfig",
    "default_workers",
    "generate_games",
    "play_round",
]
