#!/usr/bin/env python3
"""Run one or more self-play → train iterations for hold-round Hearts."""

from __future__ import annotations

import argparse
from pathlib import Path

from hearts.model import HeartsNet
from hearts.train.loop import self_play_iteration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=2, help="Self-play rounds per iteration")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pimc", action="store_true", help="Use neural-guided PIMC (slow)")
    parser.add_argument("--k", type=int, default=2, help="PIMC world samples")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Transformer width (default 128 for faster local runs; plan target 256)",
    )
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    net = HeartsNet(d_model=args.d_model, n_layers=args.layers, n_heads=4, d_ff=args.d_model * 2)

    for i in range(args.iterations):
        stats = self_play_iteration(
            net,
            n_rounds=args.rounds,
            base_seed=args.seed + i * 1000,
            use_pimc=args.pimc,
            k=args.k,
            epochs=args.epochs,
            shard_path=args.out / f"shard_{i:04d}.pt",
            checkpoint_path=args.out / f"ckpt_{i:04d}.pt",
        )
        print(f"iter={i} {stats}")


if __name__ == "__main__":
    main()
