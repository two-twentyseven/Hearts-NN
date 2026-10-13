from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

SAMPLE_KEYS = ("locations", "globals", "legal_mask", "pi", "z")


def make_shard(
    locations: np.ndarray,
    globals_: np.ndarray,
    legal_mask: np.ndarray,
    pi: np.ndarray,
    z: np.ndarray,
    meta: dict | None = None,
) -> dict:
    """Pack per-decision arrays into a shard dict of tensors.

    Shapes: locations (N,52), globals (N,G), legal_mask (N,52), pi (N,52), z (N,4).
    ``z`` is the normalized post-round score per seat, in the acting player's seat order.
    """
    n = locations.shape[0]
    for name, arr in (("globals", globals_), ("legal_mask", legal_mask), ("pi", pi), ("z", z)):
        if arr.shape[0] != n:
            raise ValueError(f"{name} has {arr.shape[0]} rows, expected {n}")
    return {
        "locations": torch.from_numpy(np.ascontiguousarray(locations)).to(torch.int8),
        "globals": torch.from_numpy(np.ascontiguousarray(globals_)).to(torch.float32),
        "legal_mask": torch.from_numpy(np.ascontiguousarray(legal_mask)).to(torch.bool),
        "pi": torch.from_numpy(np.ascontiguousarray(pi)).to(torch.float32),
        "z": torch.from_numpy(np.ascontiguousarray(z)).to(torch.float32),
        "meta": dict(meta or {}),
    }


def concat_shards(shards: list[dict]) -> dict:
    if not shards:
        raise ValueError("No shards to concatenate")
    out = {k: torch.cat([s[k] for s in shards], dim=0) for k in SAMPLE_KEYS}
    out["meta"] = {"sources": [s.get("meta", {}) for s in shards]}
    return out


def shard_size(shard: dict) -> int:
    return int(shard["z"].shape[0])


def save_shard(path: Path | str, shard: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(shard, path)
    return path


def load_shard(path: Path | str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def list_shards(directory: Path | str) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(directory.glob("*.pt"))


class ReplayBuffer:
    """Concatenation of the most recent shards, served as shuffled minibatches."""

    def __init__(self, shards: list[dict]) -> None:
        self.data = concat_shards(shards)

    @classmethod
    def from_directory(cls, directory: Path | str, window: int) -> ReplayBuffer:
        paths = list_shards(directory)
        if window > 0:
            paths = paths[-window:]
        if not paths:
            raise FileNotFoundError(f"No shards in {directory}")
        return cls([load_shard(p) for p in paths])

    def __len__(self) -> int:
        return shard_size(self.data)

    def minibatches(self, batch_size: int, rng: np.random.Generator) -> Iterator[dict[str, Tensor]]:
        n = len(self)
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = torch.from_numpy(order[start : start + batch_size])
            yield {k: self.data[k][idx] for k in SAMPLE_KEYS}
