from __future__ import annotations

import random

import torch
from torch import Tensor

from hearts.encode import encode_state
from hearts.env import HoldRoundEnv
from hearts.model import HeartsNet, legal_mask_from_cards
from hearts.pimc import pimc_select, select_from_logits
from hearts.state import Phase
from hearts.train.corpus import DecisionRecord, records_to_batch


@torch.no_grad()
def play_hold_round(
    net: HeartsNet,
    *,
    seed: int | None = None,
    use_pimc: bool = True,
    k: int = 8,
    rollout_temperature: float = 0.5,
    action_temperature: float = 0.5,
    epsilon: float = 0.1,
    rng: random.Random | None = None,
) -> dict[str, Tensor]:
    """Play one hold round; return a batched corpus sample for all decisions."""
    rng = rng or random.Random(seed)
    env = HoldRoundEnv()
    state = env.reset(seed=seed)
    records: list[DecisionRecord] = []
    net.eval()

    while state.phase is Phase.PLAYING:
        acting = state.current_player
        legal = state.legal_moves_for(acting)
        obs = encode_state(state, acting)
        mask = legal_mask_from_cards(legal)

        if use_pimc:
            result = pimc_select(
                state,
                net,
                k=k,
                rollout_temperature=rollout_temperature,
                action_temperature=action_temperature,
                epsilon=epsilon,
                rng=rng,
            )
            pi = result.pi
            card = result.action
        else:
            out = net(obs, legal_mask=mask)
            pi = torch.softmax(out.logits, dim=0)
            card = select_from_logits(out.logits, legal, action_temperature, rng)

        records.append(
            DecisionRecord(
                obs=obs,
                legal_mask=mask.clone(),
                pi_target=pi.detach().cpu(),
                acting_player=acting,
            )
        )
        state = env.step(card)

    returns = env.returns_by_player()
    return records_to_batch(records, returns)


def generate_shard(
    net: HeartsNet,
    n_rounds: int,
    *,
    base_seed: int = 0,
    use_pimc: bool = True,
    k: int = 4,
    **pimc_kwargs,
) -> list[dict[str, Tensor]]:
    """Generate n_rounds of self-play trajectories as separate batch dicts."""
    samples: list[dict[str, Tensor]] = []
    for i in range(n_rounds):
        sample = play_hold_round(
            net,
            seed=base_seed + i,
            use_pimc=use_pimc,
            k=k,
            **pimc_kwargs,
        )
        samples.append(sample)
    return samples
