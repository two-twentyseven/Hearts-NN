#!/usr/bin/env python3
"""Head-to-head evaluation for perfect-information checkpoints.

Examples
  # Does iteration 14 (with search) consistently beat random? 400 games, 1 bot vs 3 random.
  python scripts/perfect_arena.py --a artifacts/perfect/ckpt_0014.pt --sims 200 --games 400

  # What a human faces in the GUI: 3 bots vs 1 random seat.
  python scripts/perfect_arena.py --a artifacts/perfect/ckpt_0014.pt --seats-a 3 --games 400

  # Raw policy (no search) vs random.
  python scripts/perfect_arena.py --a artifacts/perfect/ckpt_0014.pt --sims 0 --games 400

  # Later checkpoint vs earlier checkpoint, both with search.
  python scripts/perfect_arena.py --a artifacts/perfect/ckpt_0014.pt --b artifacts/perfect/ckpt_0004.pt --sims 200
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from hearts.perfect.arena import AgentSpec, MatchStats, run_match


def _spec(path: str | None, sims: int, opponent_mode: str) -> AgentSpec:
    if path is None or path.lower() == "random":
        return AgentSpec(kind="random")
    if sims <= 0:
        return AgentSpec(kind="policy", checkpoint=str(Path(path)))
    return AgentSpec(kind="mcts", checkpoint=str(Path(path)), sims=sims, opponent_mode=opponent_mode)


def _report(stats: MatchStats, spec_a: AgentSpec, spec_b: AgentSpec, seconds: float) -> None:
    print()
    print(f"A = {spec_a.label}   vs   B = {spec_b.label}")
    print(f"{stats.n_games} hold rounds, {stats.seats_a_per_game} A seat(s) per round, {seconds:.0f}s")
    print(f"  mean points   A {stats.mean_a:5.2f}   B {stats.mean_b:5.2f}   (random baseline is 6.50)")
    print(f"  gap B - A     {stats.diff:+.2f} points  (95% CI +/- {stats.diff_ci95:.2f})")
    print(f"  A lowest score in round : {100 * stats.a_lowest_rate:5.1f}% of rounds")
    print(f"  A fewer points than B   : {100 * stats.a_below_b_rate:5.1f}% of rounds")
    print(f"  moons shot   A {stats.a_moons}   B {stats.b_moons}")
    lo = stats.diff - stats.diff_ci95
    if lo > 0:
        print(f"  verdict: A is better than B (gap is at least {lo:.2f} points at 95% confidence)")
    elif stats.diff + stats.diff_ci95 < 0:
        print("  verdict: A is WORSE than B at 95% confidence")
    else:
        print("  verdict: not separable at 95% confidence; play more games")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="Checkpoint for agent A (or 'random')")
    p.add_argument("--b", default="random", help="Checkpoint for agent B, or 'random' (default)")
    p.add_argument("--sims", type=int, default=200, help="MCTS simulations for A (0 = raw policy, no search)")
    p.add_argument("--b-sims", type=int, default=None, help="MCTS simulations for B when B is a checkpoint (default: --sims)")
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--seats-a", type=int, default=1, choices=[1, 2, 3], help="How many seats A occupies per round")
    p.add_argument("--opponent-mode", choices=["paranoid", "self_interest"], default="paranoid")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--workers", type=int, default=0, help="Processes (0 = cpu_count-1)")
    p.add_argument("--json", type=Path, default=None, help="Also write per-round outcomes to this file")
    args = p.parse_args()

    spec_a = _spec(args.a, args.sims, args.opponent_mode)
    spec_b = _spec(args.b, args.b_sims if args.b_sims is not None else args.sims, args.opponent_mode)
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    print(f"playing {args.games} rounds: {spec_a.label} vs {spec_b.label} ({workers} workers)")
    t0 = time.time()
    stats = run_match(
        spec_a, spec_b, args.games,
        seats_a_per_game=args.seats_a, base_seed=args.seed, workers=workers, progress=True,
    )
    _report(stats, spec_a, spec_b, time.time() - t0)
    if args.json is not None:
        payload = {
            "a": spec_a.__dict__, "b": spec_b.__dict__, "seats_a_per_game": stats.seats_a_per_game,
            "n_games": stats.n_games, "mean_a": stats.mean_a, "mean_b": stats.mean_b,
            "diff": stats.diff, "diff_ci95": stats.diff_ci95,
            "a_lowest_rate": stats.a_lowest_rate, "a_below_b_rate": stats.a_below_b_rate,
            "a_moons": stats.a_moons, "b_moons": stats.b_moons,
            "outcomes": [{"seed": o.seed, "seats_a": list(o.seats_a), "scores": o.scores} for o in stats.outcomes],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
