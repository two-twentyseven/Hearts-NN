from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW

from hearts.perfect.corpus import ReplayBuffer
from hearts.perfect.encode import POINTS_PER_ROUND
from hearts.perfect.model import PerfectNet


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    c_v: float = 1.0
    grad_clip: float = 1.0


def policy_value_loss(
    net: PerfectNet,
    batch: dict[str, Tensor],
    *,
    c_v: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Cross-entropy(pi_target, policy) over legal cards + MSE(values, z) over the 4 seats."""
    device = net.device
    locations = batch["locations"].to(device=device, dtype=torch.long)
    globals_ = batch["globals"].to(device)
    mask = batch["legal_mask"].to(device)
    pi = batch["pi"].to(device)
    z = batch["z"].to(device)

    out = net(locations, globals_, legal_mask=mask)
    log_probs = torch.log_softmax(out.logits, dim=-1)
    log_probs = torch.where(mask, log_probs, torch.zeros_like(log_probs))
    policy_loss = -(pi * log_probs).sum(dim=-1).mean()
    value_loss = nn.functional.mse_loss(out.values, z)
    loss = policy_loss + c_v * value_loss

    with torch.no_grad():
        # Value MAE in round points for the acting seat (index 0) and all seats.
        abs_err_points = (out.values - z).abs() * (POINTS_PER_ROUND / 2.0)
        target_entropy = -(pi * torch.log(pi.clamp_min(1e-12))).sum(dim=-1).mean()
        top1 = (out.logits.argmax(dim=-1) == pi.argmax(dim=-1)).float().mean()
    return loss, {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "value_mae_points_self": float(abs_err_points[:, 0].mean()),
        "value_mae_points_all": float(abs_err_points.mean()),
        "policy_top1": float(top1),
        "target_entropy": float(target_entropy),
    }


def make_optimizer(net: PerfectNet, cfg: TrainConfig) -> AdamW:
    return AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def train_on_buffer(
    net: PerfectNet,
    buffer: ReplayBuffer,
    cfg: TrainConfig,
    *,
    optimizer: AdamW | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Run ``cfg.epochs`` passes of minibatch SGD over the buffer; return mean stats of the last epoch."""
    rng = rng if rng is not None else np.random.default_rng()
    opt = optimizer or make_optimizer(net, cfg)
    net.train()
    last: dict[str, float] = {}
    for _ in range(cfg.epochs):
        sums: dict[str, float] = {}
        steps = 0
        for batch in buffer.minibatches(cfg.batch_size, rng):
            opt.zero_grad(set_to_none=True)
            loss, stats = policy_value_loss(net, batch, c_v=cfg.c_v)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            for k, v in stats.items():
                sums[k] = sums.get(k, 0.0) + v
            steps += 1
        last = {k: v / max(steps, 1) for k, v in sums.items()}
        last["steps"] = float(steps)
        last["samples"] = float(len(buffer))
    net.eval()
    return last
