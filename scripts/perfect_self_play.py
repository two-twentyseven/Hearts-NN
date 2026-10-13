#!/usr/bin/env python3
"""Perfect-information Hearts: self-play -> corpus -> train loop.

Each iteration:
  1. play --games hold rounds with the current net + paranoid MCTS (--sims per move)
  2. save the decisions as a corpus shard under <out>/shards/
  3. train on the last --replay-window shards
  4. save <out>/ckpt_XXXX.pt (and <out>/latest.pt)
  5. optionally play --eval-games against a legal-random opponent

Re-run with --resume <out>/latest.pt to keep cycling.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from hearts.perfect.arena import MCTSAgent, PolicyAgent, RandomAgent, evaluate
from hearts.perfect.corpus import ReplayBuffer, save_shard
from hearts.perfect.mcts import MCTSConfig
from hearts.perfect.model import NetConfig, PerfectNet, load_checkpoint, save_checkpoint
from hearts.perfect.self_play import SelfPlayConfig, default_workers, generate_games
from hearts.perfect.train import TrainConfig, make_optimizer, train_on_buffer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("artifacts/perfect"))
    p.add_argument("--resume", type=Path, default=None, help="Checkpoint to continue from")
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    # self-play
    p.add_argument("--games", type=int, default=64, help="Self-play rounds per iteration")
    p.add_argument("--sims", type=int, default=100, help="MCTS simulations per move")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--dirichlet-alpha", type=float, default=0.6)
    p.add_argument("--dirichlet-eps", type=float, default=0.25)
    p.add_argument("--opponent-mode", choices=["paranoid", "self_interest"], default="paranoid")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temperature-cutoff", type=int, default=7, help="Trick number from which play is greedy")
    p.add_argument("--workers", type=int, default=0, help="Self-play processes (0 = cpu_count-1)")
    # training
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--c-v", type=float, default=1.0)
    p.add_argument("--replay-window", type=int, default=10, help="Train on the last N shards (0 = all)")
    # model
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--blocks", type=int, default=4)
    # eval
    p.add_argument("--eval-games", type=int, default=0, help="Arena games vs random after each iteration")
    p.add_argument("--eval-sims", type=int, default=50)
    p.add_argument("--threads", type=int, default=0, help="torch threads for training (0 = default)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    shard_dir = args.out / "shards"
    log_path = args.out / "log.jsonl"

    start_iter = 0
    if args.resume is not None:
        net, payload = load_checkpoint(args.resume)
        start_iter = int(payload.get("iteration", -1)) + 1
        print(f"resumed {args.resume} at iteration {start_iter} ({net.num_parameters():,} params)")
    else:
        net = PerfectNet(NetConfig(d_model=args.d_model, n_blocks=args.blocks))
        print(f"new PerfectNet ({net.num_parameters():,} params)")

    mcts_cfg = MCTSConfig(
        n_simulations=args.sims,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_eps=args.dirichlet_eps,
        opponent_mode=args.opponent_mode,
    )
    play_cfg = SelfPlayConfig(temperature=args.temperature, temperature_cutoff_trick=args.temperature_cutoff)
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        c_v=args.c_v,
    )
    optimizer = make_optimizer(net, train_cfg)
    workers = args.workers if args.workers > 0 else default_workers()
    rng = np.random.default_rng(args.seed)

    for it in range(start_iter, start_iter + args.iterations):
        t0 = time.time()
        base_seed = args.seed + it * 100_000
        print(f"[iter {it}] self-play: {args.games} games x {args.sims} sims, {workers} workers")
        shard, scores = generate_games(
            net,
            args.games,
            base_seed=base_seed,
            mcts_cfg=mcts_cfg,
            play_cfg=play_cfg,
            workers=workers,
            progress=False,
        )
        shard["meta"]["iteration"] = it
        shard_path = save_shard(shard_dir / f"iter_{it:04d}.pt", shard)
        t1 = time.time()
        # Per-game std of the four scores: 0 means points were split evenly, ~11 means one seat ate everything.
        spread = float(np.mean([np.std(s) for s in scores]))
        moons = sum(1 for s in scores if sum(s) == 78)
        print(f"[iter {it}] saved {shard['z'].shape[0]} decisions -> {shard_path} "
              f"(score spread {spread:.2f}, moons {moons}, {t1 - t0:.0f}s)")

        buffer = ReplayBuffer.from_directory(shard_dir, window=args.replay_window)
        stats = train_on_buffer(net, buffer, train_cfg, optimizer=optimizer, rng=rng)
        t2 = time.time()
        print(f"[iter {it}] train on {len(buffer)} samples: "
              f"loss={stats['loss']:.4f} policy={stats['policy_loss']:.4f} value={stats['value_loss']:.4f} "
              f"top1={stats['policy_top1']:.3f} vmae_self={stats['value_mae_points_self']:.2f}pts ({t2 - t1:.0f}s)")

        extra = {"iteration": it, "train_stats": stats, "mcts_config": mcts_cfg.__dict__, "args": vars(args) | {"out": str(args.out), "resume": str(args.resume)}}
        save_checkpoint(args.out / f"ckpt_{it:04d}.pt", net, extra)
        save_checkpoint(args.out / "latest.pt", net, extra)

        eval_stats: dict = {}
        if args.eval_games > 0:
            eval_cfg = MCTSConfig(n_simulations=args.eval_sims, dirichlet_eps=0.0, opponent_mode=args.opponent_mode)
            vs_random_policy = evaluate(PolicyAgent(net), RandomAgent(), args.eval_games, base_seed=base_seed + 50_000)
            vs_random_mcts = evaluate(MCTSAgent(net, eval_cfg), RandomAgent(), args.eval_games, base_seed=base_seed + 50_000)
            eval_stats = {
                "policy_vs_random": (vs_random_policy.mean_points_a, vs_random_policy.mean_points_b),
                "mcts_vs_random": (vs_random_mcts.mean_points_a, vs_random_mcts.mean_points_b),
            }
            print(f"[iter {it}] eval: policy {vs_random_policy.mean_points_a:.2f} vs random {vs_random_policy.mean_points_b:.2f} | "
                  f"mcts {vs_random_mcts.mean_points_a:.2f} vs random {vs_random_mcts.mean_points_b:.2f} ({time.time() - t2:.0f}s)")

        with log_path.open("a") as f:
            f.write(json.dumps({
                "iteration": it,
                "decisions": int(shard["z"].shape[0]),
                "self_play_score_spread": spread,
                "moons": moons,
                "train": stats,
                "eval": eval_stats,
                "seconds": {"self_play": t1 - t0, "train": t2 - t1, "total": time.time() - t0},
            }) + "\n")


if __name__ == "__main__":
    main()
