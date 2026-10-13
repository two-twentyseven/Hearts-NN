from __future__ import annotations

import torch

from hearts.model import HeartsNet, observation_to_tensors
from hearts.train.loop import self_play_iteration, train_on_samples
from hearts.train.loss import policy_value_loss
from hearts.train.self_play import play_hold_round
from hearts.env import HoldRoundEnv
from hearts.encode import encode_state
from hearts.model import legal_mask_from_cards
from hearts.state import DECK_SIZE


def test_policy_value_loss_backprops() -> None:
    env = HoldRoundEnv()
    state = env.reset(seed=3)
    acting = state.current_player
    obs = encode_state(state, acting)
    mask = legal_mask_from_cards(state.legal_moves_for(acting))
    tensors = observation_to_tensors(obs)
    pi = torch.zeros(DECK_SIZE)
    legal_ids = mask.nonzero(as_tuple=False).view(-1)
    pi[legal_ids] = 1.0 / len(legal_ids)

    batch = {
        **{k: v.unsqueeze(0) for k, v in tensors.items()},
        "legal_mask": mask.unsqueeze(0),
        "pi_target": pi.unsqueeze(0),
        "z": torch.tensor([0.0]),
        "acting_player": torch.tensor([acting]),
    }
    net = HeartsNet(d_model=32, n_layers=1, n_heads=4, d_ff=64)
    net.train()
    loss, stats = policy_value_loss(net, batch)
    assert torch.isfinite(loss)
    loss.backward()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    assert grad_ok
    assert "policy_loss" in stats and "value_loss" in stats


def test_self_play_iteration_policy_only() -> None:
    net = HeartsNet(d_model=32, n_layers=1, n_heads=4, d_ff=64)
    stats = self_play_iteration(
        net,
        n_rounds=1,
        base_seed=11,
        use_pimc=False,
        epochs=1,
    )
    assert torch.isfinite(torch.tensor(stats["loss"]))


def test_play_hold_round_policy_only_batch_shapes() -> None:
    net = HeartsNet(d_model=32, n_layers=1, n_heads=4, d_ff=64)
    batch = play_hold_round(net, seed=5, use_pimc=False, action_temperature=0.0)
    assert batch["z"].ndim == 1
    assert batch["pi_target"].shape[1] == DECK_SIZE
    assert batch["legal_mask"].shape == batch["pi_target"].shape
    assert batch["in_hand"].shape[0] == batch["z"].shape[0]
