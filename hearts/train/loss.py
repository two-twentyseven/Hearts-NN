from __future__ import annotations

import torch
from torch import Tensor, nn

from hearts.model import HeartsNet


def policy_value_loss(
    net: HeartsNet,
    batch: dict[str, Tensor],
    *,
    c_v: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Masked KL(policy || pi_target) + MSE(value, z)."""
    device = next(net.parameters()).device
    # Split batch into per-sample forwards (network is unbatched).
    n = batch["z"].shape[0]
    policy_losses: list[Tensor] = []
    value_preds: list[Tensor] = []
    for i in range(n):
        obs = {k: batch[k][i] for k in batch if k not in ("legal_mask", "pi_target", "z", "acting_player")}
        mask = batch["legal_mask"][i].to(device)
        pi_target = batch["pi_target"][i].to(device)
        out = net(obs, legal_mask=mask)
        log_probs = torch.log_softmax(out.logits, dim=0)
        # Avoid 0 * -inf = NaN on illegal slots.
        safe_log = torch.where(mask, log_probs, torch.zeros_like(log_probs))
        policy_losses.append(-(pi_target * safe_log).sum())
        value_preds.append(out.value)

    policy_loss = torch.stack(policy_losses).mean()
    values = torch.stack(value_preds)
    z = batch["z"].to(device)
    value_loss = nn.functional.mse_loss(values, z)
    loss = policy_loss + c_v * value_loss
    return loss, {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
    }
