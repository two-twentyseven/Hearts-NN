from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor, nn

from hearts.perfect.encode import GLOBAL_DIM, NUM_LOCATIONS, PerfectObs
from hearts.rules import NUM_PLAYERS
from hearts.state import DECK_SIZE


@dataclass
class PerfectOutput:
    logits: Tensor  # (B, 52); illegal entries are -inf when a mask is given
    values: Tensor  # (B, 4) in [-1, 1]; predicted normalized post-round score per relative seat


@dataclass(frozen=True)
class NetConfig:
    d_model: int = 512
    n_blocks: int = 4
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class ResidualBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.ff(self.norm(x))


class PerfectNet(nn.Module):
    """Batched residual MLP over the perfect-information encoding.

    Input: ``locations`` (B, 52) long in 0..11 and ``globals`` (B, GLOBAL_DIM).
    Output: 52 policy logits and a 4-vector value (one entry per relative seat).
    """

    def __init__(self, config: NetConfig | None = None, **overrides) -> None:
        super().__init__()
        cfg = config or NetConfig()
        if overrides:
            cfg = NetConfig(**{**cfg.to_dict(), **overrides})
        self.config = cfg
        in_dim = DECK_SIZE * NUM_LOCATIONS + GLOBAL_DIM
        self.input = nn.Linear(in_dim, cfg.d_model)
        self.blocks = nn.ModuleList(
            [ResidualBlock(cfg.d_model, cfg.dropout) for _ in range(cfg.n_blocks)]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.policy_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, DECK_SIZE),
        )
        self.value_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, NUM_PLAYERS),
            nn.Tanh(),
        )

    @property
    def device(self) -> torch.device:
        return self.input.weight.device

    def forward(
        self,
        locations: Tensor,
        globals_: Tensor,
        legal_mask: Tensor | None = None,
    ) -> PerfectOutput:
        squeeze = locations.dim() == 1
        if squeeze:
            locations = locations.unsqueeze(0)
            globals_ = globals_.unsqueeze(0)
            if legal_mask is not None:
                legal_mask = legal_mask.unsqueeze(0)
        device = self.device
        locations = locations.to(device=device, dtype=torch.long)
        globals_ = globals_.to(device=device, dtype=torch.float32)
        one_hot = nn.functional.one_hot(locations, NUM_LOCATIONS).to(torch.float32)
        x = torch.cat([one_hot.flatten(1), globals_], dim=-1)
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        logits = self.policy_head(h)
        values = self.value_head(h)
        if legal_mask is not None:
            mask = legal_mask.to(device=device, dtype=torch.bool)
            logits = logits.masked_fill(~mask, float("-inf"))
        if squeeze:
            logits = logits.squeeze(0)
            values = values.squeeze(0)
        return PerfectOutput(logits=logits, values=values)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def obs_to_tensors(obs: PerfectObs) -> tuple[Tensor, Tensor]:
    return (
        torch.from_numpy(np.ascontiguousarray(obs.locations)).to(torch.long),
        torch.from_numpy(np.ascontiguousarray(obs.globals)).to(torch.float32),
    )


def stack_obs(obs_list: list[PerfectObs]) -> tuple[Tensor, Tensor]:
    locs = torch.from_numpy(np.stack([o.locations for o in obs_list])).to(torch.long)
    globs = torch.from_numpy(np.stack([o.globals for o in obs_list])).to(torch.float32)
    return locs, globs


@torch.inference_mode()
def predict(
    net: PerfectNet,
    obs_list: list[PerfectObs],
    legal_masks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a batch of positions.

    Returns ``(priors, values)`` where ``priors`` is (B, 52) softmax over legal
    cards (zeros elsewhere) and ``values`` is (B, 4) in the acting player's seat order.
    """
    locs, globs = stack_obs(obs_list)
    mask = torch.from_numpy(np.stack(legal_masks))
    out = net(locs, globs, legal_mask=mask)
    priors = torch.softmax(out.logits, dim=-1)
    priors = torch.nan_to_num(priors, nan=0.0)
    return priors.cpu().numpy(), out.values.cpu().numpy()


def save_checkpoint(path, net: PerfectNet, extra: dict | None = None) -> None:
    payload = {"model": net.state_dict(), "net_config": net.config.to_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, map_location="cpu") -> tuple[PerfectNet, dict]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    net = PerfectNet(NetConfig(**payload["net_config"]))
    net.load_state_dict(payload["model"])
    return net, payload
