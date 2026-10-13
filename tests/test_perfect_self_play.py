from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from hearts.perfect.arena import MCTSAgent, PolicyAgent, RandomAgent, evaluate
from hearts.perfect.corpus import ReplayBuffer, load_shard, save_shard, shard_size
from hearts.perfect.encode import normalize_scores, rotate_to_relative
from hearts.perfect.mcts import MCTSConfig
from hearts.perfect.model import NetConfig, PerfectNet
from hearts.perfect.self_play import SelfPlayConfig, generate_games, play_round
from hearts.perfect.train import TrainConfig, policy_value_loss, train_on_buffer
from hearts.state import DECK_SIZE


def _net() -> PerfectNet:
    return PerfectNet(NetConfig(d_model=64, n_blocks=1))


def test_play_round_records_and_targets() -> None:
    net = _net()
    data = play_round(net, MCTSConfig(n_simulations=8), SelfPlayConfig(), seed=4)
    shard = data.shard
    assert shard_size(shard) == DECK_SIZE
    assert shard["locations"].shape == (DECK_SIZE, DECK_SIZE)
    assert shard["pi"].shape == (DECK_SIZE, DECK_SIZE)
    assert shard["z"].shape == (DECK_SIZE, 4)
    pi = shard["pi"]
    mask = shard["legal_mask"]
    assert torch.allclose(pi.sum(dim=1), torch.ones(DECK_SIZE))
    assert (pi[~mask] == 0).all()
    # All four seats contribute equally: 13 decisions each.
    assert sorted(data.actors) == sorted([0, 1, 2, 3] * 13)
    # Every row's value target is the true round result rotated to that row's actor.
    z_abs = normalize_scores(data.round_scores)
    for i, actor in enumerate(data.actors):
        assert np.allclose(shard["z"][i].numpy(), rotate_to_relative(z_abs, actor))
    assert sum(data.round_scores) in (26, 78)  # normal or shot-the-moon totals


def test_generate_games_merges_and_saves(tmp_path: Path) -> None:
    net = _net()
    shard, scores = generate_games(net, 2, base_seed=10, mcts_cfg=MCTSConfig(n_simulations=4), workers=1)
    assert shard_size(shard) == 2 * DECK_SIZE
    assert len(scores) == 2
    path = save_shard(tmp_path / "shards" / "iter_0000.pt", shard)
    loaded = load_shard(path)
    assert shard_size(loaded) == 2 * DECK_SIZE
    buffer = ReplayBuffer.from_directory(tmp_path / "shards", window=5)
    assert len(buffer) == 2 * DECK_SIZE
    batches = list(buffer.minibatches(40, np.random.default_rng(0)))
    assert sum(b["z"].shape[0] for b in batches) == 2 * DECK_SIZE


def test_train_step_reduces_loss(tmp_path: Path) -> None:
    net = _net()
    shard, _ = generate_games(net, 1, base_seed=3, mcts_cfg=MCTSConfig(n_simulations=4), workers=1)
    save_shard(tmp_path / "s" / "iter_0000.pt", shard)
    buffer = ReplayBuffer.from_directory(tmp_path / "s", window=0)
    batch = {k: buffer.data[k] for k in ("locations", "globals", "legal_mask", "pi", "z")}
    net.eval()
    with torch.no_grad():
        before, stats = policy_value_loss(net, batch)
    assert torch.isfinite(before)
    assert set(stats) >= {"policy_loss", "value_loss", "policy_top1", "value_mae_points_self"}
    train_on_buffer(net, buffer, TrainConfig(epochs=8, batch_size=64, lr=1e-3), rng=np.random.default_rng(0))
    with torch.no_grad():
        after, _ = policy_value_loss(net, batch)
    assert float(after) < float(before)


def test_run_match_stats(tmp_path: Path) -> None:
    from hearts.perfect.arena import AgentSpec, run_match
    from hearts.perfect.model import save_checkpoint

    ckpt = tmp_path / "c.pt"
    save_checkpoint(ckpt, _net(), {"iteration": 0})
    stats = run_match(
        AgentSpec(kind="mcts", checkpoint=str(ckpt), sims=4),
        AgentSpec(kind="random"),
        6,
        seats_a_per_game=3,
        base_seed=1,
        workers=1,
    )
    assert stats.n_games == 6 and stats.seats_a_per_game == 3
    assert all(len(o.seats_a) == 3 for o in stats.outcomes)
    # Seats rotate so every seat hosts A across games.
    assert {s for o in stats.outcomes for s in o.seats_a} == {0, 1, 2, 3}
    assert 0.0 <= stats.a_lowest_rate <= 1.0
    assert abs(stats.diff - (stats.mean_b - stats.mean_a)) < 1e-9
    assert all(sum(o.scores) in (26, 78) for o in stats.outcomes)
    policy_stats = run_match(AgentSpec(kind="policy", checkpoint=str(ckpt)), AgentSpec(kind="random"), 2, workers=1)
    assert policy_stats.n_games == 2


def test_arena_runs() -> None:
    net = _net()
    res = evaluate(PolicyAgent(net), RandomAgent(), 2, base_seed=1)
    assert res.n_games == 2 and len(res.per_game) == 2
    res2 = evaluate(MCTSAgent(net, MCTSConfig(n_simulations=4, dirichlet_eps=0.0)), RandomAgent(), 1)
    assert res2.n_games == 1


def test_cli_one_iteration(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "perfect_self_play.py"
    out = tmp_path / "run"
    cmd = [
        sys.executable, str(script), "--out", str(out), "--iterations", "1", "--games", "1", "--sims", "4",
        "--workers", "1", "--epochs", "1", "--batch-size", "32", "--d-model", "32", "--blocks", "1",
        "--eval-games", "1", "--eval-sims", "2",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert (out / "latest.pt").exists()
    assert (out / "ckpt_0000.pt").exists()
    assert (out / "shards" / "iter_0000.pt").exists()
    assert (out / "log.jsonl").exists()
    # Resume continues from iteration 1.
    proc2 = subprocess.run(cmd + ["--resume", str(out / "latest.pt")], capture_output=True, text=True, timeout=600)
    assert proc2.returncode == 0, proc2.stderr
    assert (out / "ckpt_0001.pt").exists()
