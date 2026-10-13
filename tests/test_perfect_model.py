from __future__ import annotations

import numpy as np
import torch

from hearts.perfect.encode import encode_perfect, legal_mask_for
from hearts.perfect.game import new_hold_round
from hearts.perfect.model import (
    NetConfig,
    PerfectNet,
    load_checkpoint,
    predict,
    save_checkpoint,
    stack_obs,
)
from hearts.state import DECK_SIZE


def _positions(n: int):
    obs_list, masks = [], []
    for seed in range(n):
        state = new_hold_round(seed=seed)
        for _ in range(seed % 7):
            legal = state.legal_moves_for(state.current_player)
            state = state.play_card(state.current_player, legal[0])
        obs_list.append(encode_perfect(state, state.current_player))
        masks.append(legal_mask_for(state))
    return obs_list, masks


def test_batched_forward_shapes_and_mask() -> None:
    obs_list, masks = _positions(5)
    net = PerfectNet(NetConfig(d_model=64, n_blocks=2)).eval()
    locs, globs = stack_obs(obs_list)
    mask = torch.from_numpy(np.stack(masks))
    with torch.no_grad():
        out = net(locs, globs, legal_mask=mask)
    assert out.logits.shape == (5, DECK_SIZE)
    assert out.values.shape == (5, 4)
    assert torch.isfinite(out.logits[mask]).all()
    assert torch.isneginf(out.logits[~mask]).all()
    assert (out.values.abs() <= 1.0).all()

    single = net(locs[0], globs[0], legal_mask=mask[0])
    assert single.logits.shape == (DECK_SIZE,)
    assert single.values.shape == (4,)


def test_predict_returns_legal_distribution() -> None:
    obs_list, masks = _positions(3)
    net = PerfectNet(NetConfig(d_model=64, n_blocks=1))
    priors, values = predict(net, obs_list, masks)
    assert priors.shape == (3, DECK_SIZE) and values.shape == (3, 4)
    for p, m in zip(priors, masks):
        assert np.isclose(p.sum(), 1.0)
        assert (p[~m] == 0).all()


def test_gradients_flow() -> None:
    obs_list, masks = _positions(4)
    net = PerfectNet(NetConfig(d_model=64, n_blocks=2)).train()
    locs, globs = stack_obs(obs_list)
    out = net(locs, globs)
    (out.logits.sum() + out.values.sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_checkpoint_round_trip(tmp_path) -> None:
    net = PerfectNet(NetConfig(d_model=32, n_blocks=1))
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, net, {"iteration": 3})
    loaded, payload = load_checkpoint(path)
    assert payload["iteration"] == 3
    assert loaded.config == net.config
    for a, b in zip(net.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
