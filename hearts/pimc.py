from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from hearts.cards import Card
from hearts.encode import encode_state
from hearts.env import card_from_id, normalize_round_return
from hearts.model import HeartsNet, legal_mask_from_cards
from hearts.rules import NUM_PLAYERS, finalize_round_scores
from hearts.state import DECK_SIZE, GameState, Phase


@dataclass
class PIMCResult:
    pi: Tensor  # (52,) soft target over legal cards
    action: Card
    mean_returns: Tensor  # (52,) NaN for illegal


def remaining_unknown_cards(state: GameState, acting_player: int) -> list[Card]:
    """Cards not in acting hand and not yet played (publicly)."""
    played_ids: set[int] = set()
    for by_card in state.card_play_tricks:
        for card_id, trick_frac in enumerate(by_card):
            if trick_frac > 0.0:
                played_ids.add(card_id)
    # Cards on the current trick are already recorded in card_play_tricks.
    known = {card.card_id for card in state.hands[acting_player]} | played_ids
    return [card_from_id(i) for i in range(DECK_SIZE) if i not in known]


def sample_opponent_hands(
    state: GameState,
    acting_player: int,
    rng: random.Random,
) -> list[list[Card]]:
    """Partition unknown cards into the three opponent hands matching current sizes."""
    unknown = remaining_unknown_cards(state, acting_player)
    sizes = [len(state.hands[p]) for p in range(NUM_PLAYERS)]
    need = sum(sizes[p] for p in range(NUM_PLAYERS) if p != acting_player)
    if len(unknown) < need:
        raise RuntimeError(
            f"Not enough unknown cards to fill opponent hands ({len(unknown)} < {need})"
        )
    rng.shuffle(unknown)
    # Full deals: len(unknown) == need. Truncated test states may have extras.
    unknown = unknown[:need]
    hands: list[list[Card]] = [[] for _ in range(NUM_PLAYERS)]
    hands[acting_player] = list(state.hands[acting_player])
    idx = 0
    for player in range(NUM_PLAYERS):
        if player == acting_player:
            continue
        take = sizes[player]
        hands[player] = unknown[idx : idx + take]
        idx += take
    return hands


def apply_sampled_hands(state: GameState, hands: list[list[Card]]) -> GameState:
    return replace(state, hands=deepcopy(hands))


def select_from_logits(
    logits: Tensor,
    legal_cards: list[Card],
    temperature: float,
    rng: random.Random,
) -> Card:
    """Sample a legal card from masked logits (greedy if temperature <= 0)."""
    if not legal_cards:
        raise ValueError("No legal cards")
    legal_ids = [c.card_id for c in legal_cards]
    legal_logits = logits[legal_ids]
    if temperature <= 0:
        best = int(torch.argmax(legal_logits).item())
        return legal_cards[best]
    probs = torch.softmax(legal_logits / temperature, dim=0)
    choice = int(torch.multinomial(probs, 1).item())
    return legal_cards[choice]


@torch.no_grad()
def policy_rollout(
    state: GameState,
    net: HeartsNet,
    temperature: float,
    rng: random.Random,
) -> list[int]:
    """Play out the rest of the round with the network; return round scores."""
    net.eval()
    s = state
    for _ in range(DECK_SIZE + 4):
        if s.phase is not Phase.PLAYING:
            break
        player = s.current_player
        legal = s.legal_moves_for(player)
        if not legal:
            # Truncated deals (tests) or empty hand: score what was captured.
            return finalize_round_scores(s.tricks_won, s.config)
        obs = encode_state(s, player)
        mask = legal_mask_from_cards(legal)
        out = net(obs, legal_mask=mask)
        card = select_from_logits(out.logits, legal, temperature, rng)
        s = s.play_card(player, card)
    if s.last_round_scores is not None:
        return s.last_round_scores
    return finalize_round_scores(s.tricks_won, s.config)


@torch.no_grad()
def pimc_select(
    state: GameState,
    net: HeartsNet,
    *,
    k: int = 8,
    rollout_temperature: float = 0.5,
    action_temperature: float = 0.5,
    epsilon: float = 0.1,
    rng: random.Random | None = None,
) -> PIMCResult:
    """Neural-guided PIMC: sample worlds, rollout each legal first action, soft target."""
    rng = rng or random.Random()
    acting = state.current_player
    legal = state.legal_moves_for(acting)
    if not legal:
        raise ValueError("No legal moves")

    mean_returns = torch.full((DECK_SIZE,), float("nan"))
    scores = torch.zeros(DECK_SIZE)
    counts = torch.zeros(DECK_SIZE)

    for _ in range(k):
        hands = sample_opponent_hands(state, acting, rng)
        world = apply_sampled_hands(state, hands)
        for card in legal:
            after = world.play_card(acting, card)
            if after.phase is Phase.PLAYING:
                round_scores = policy_rollout(after, net, rollout_temperature, rng)
            else:
                round_scores = after.last_round_scores
                if round_scores is None:
                    raise RuntimeError("Expected round scores after terminal play")
            z = normalize_round_return(round_scores[acting])
            scores[card.card_id] += z
            counts[card.card_id] += 1.0

    for card in legal:
        mean_returns[card.card_id] = scores[card.card_id] / counts[card.card_id]

    legal_ids = [c.card_id for c in legal]
    legal_means = mean_returns[legal_ids]
    # Softmax over mean returns (higher z = better).
    if action_temperature <= 0:
        pi_legal = torch.zeros(len(legal_ids))
        pi_legal[int(torch.argmax(legal_means).item())] = 1.0
    else:
        pi_legal = torch.softmax(legal_means / action_temperature, dim=0)

    pi = torch.zeros(DECK_SIZE)
    for i, card_id in enumerate(legal_ids):
        pi[card_id] = pi_legal[i]

    # Optional mix toward raw network policy for exploration.
    if epsilon > 0:
        obs = encode_state(state, acting)
        mask = legal_mask_from_cards(legal)
        raw = torch.softmax(net(obs, legal_mask=mask).logits, dim=0)
        # Zero illegal already via mask → -inf softmax.
        pi = (1.0 - epsilon) * pi + epsilon * raw
        pi = pi / pi.sum()

    action = select_from_logits(torch.log(pi.clamp_min(1e-12)), legal, action_temperature, rng)
    return PIMCResult(pi=pi, action=action, mean_returns=mean_returns)
