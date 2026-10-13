from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from hearts.model import observation_to_tensors
from hearts.encode import EncodedObservation


class DecisionRecord:
    __slots__ = ("obs", "legal_mask", "pi_target", "acting_player")

    def __init__(
        self,
        obs: EncodedObservation,
        legal_mask: Tensor,
        pi_target: Tensor,
        acting_player: int,
    ) -> None:
        self.obs = obs
        self.legal_mask = legal_mask
        self.pi_target = pi_target
        self.acting_player = acting_player


def records_to_batch(
    records: list[DecisionRecord],
    returns_by_player: list[float],
) -> dict[str, Tensor]:
    """Stack trajectory decisions; each sample gets z for its acting player."""
    if not records:
        raise ValueError("Empty trajectory")
    obs_keys = observation_to_tensors(records[0].obs).keys()
    batch_obs: dict[str, list[Tensor]] = {k: [] for k in obs_keys}
    masks: list[Tensor] = []
    pis: list[Tensor] = []
    zs: list[float] = []
    actors: list[int] = []
    for rec in records:
        tensors = observation_to_tensors(rec.obs)
        for k in obs_keys:
            batch_obs[k].append(tensors[k])
        masks.append(rec.legal_mask)
        pis.append(rec.pi_target)
        zs.append(returns_by_player[rec.acting_player])
        actors.append(rec.acting_player)
    return {
        **{k: torch.stack(v, dim=0) for k, v in batch_obs.items()},
        "legal_mask": torch.stack(masks, dim=0),
        "pi_target": torch.stack(pis, dim=0),
        "z": torch.tensor(zs, dtype=torch.float32),
        "acting_player": torch.tensor(actors, dtype=torch.long),
    }


def save_shard(path: Path | str, samples: list[dict[str, Tensor]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, path)


def load_shard(path: Path | str) -> list[dict[str, Tensor]]:
    return torch.load(path, map_location="cpu", weights_only=False)


def concat_batches(batches: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    keys = batches[0].keys()
    return {k: torch.cat([b[k] for b in batches], dim=0) for k in keys}
