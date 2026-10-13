from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW

from hearts.model import HeartsNet
from hearts.train.corpus import concat_batches, load_shard, save_shard
from hearts.train.loss import policy_value_loss
from hearts.train.self_play import generate_shard


def train_on_samples(
    net: HeartsNet,
    samples: list[dict],
    *,
    epochs: int = 1,
    lr: float = 1e-4,
    c_v: float = 1.0,
) -> dict[str, float]:
    """Run a few epochs of SGD over concatenated self-play samples."""
    if not samples:
        raise ValueError("No samples to train on")
    batch = concat_batches(samples)
    opt = AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    net.train()
    last: dict[str, float] = {}
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss, stats = policy_value_loss(net, batch, c_v=c_v)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        last = stats
    return last


def self_play_iteration(
    net: HeartsNet,
    *,
    n_rounds: int = 2,
    base_seed: int = 0,
    use_pimc: bool = False,
    k: int = 2,
    epochs: int = 1,
    shard_path: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
) -> dict[str, float]:
    """Generate a corpus shard, train, optionally save checkpoint."""
    samples = generate_shard(
        net,
        n_rounds,
        base_seed=base_seed,
        use_pimc=use_pimc,
        k=k,
    )
    if shard_path is not None:
        save_shard(shard_path, samples)
    stats = train_on_samples(net, samples, epochs=epochs)
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": net.state_dict(), "stats": stats}, path)
    return stats
