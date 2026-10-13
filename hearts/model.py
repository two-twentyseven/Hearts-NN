from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from hearts.encode import EncodedObservation
from hearts.rules import CARDS_PER_HAND, NUM_PLAYERS
from hearts.state import DECK_SIZE

SEAT_NONE = NUM_PLAYERS  # embedding index 4
TRICK_NONE = CARDS_PER_HAND  # embedding index 13
SUIT_NONE = 4  # embedding index for no suit led
POSITION_NONE = NUM_PLAYERS  # embedding index 4


def _map_none(indices: Tensor, none_id: int) -> Tensor:
    """Map NONE_INDEX (-1) to ``none_id`` for embedding lookup."""
    return torch.where(indices < 0, torch.full_like(indices, none_id), indices)


@dataclass
class PolicyValueOutput:
    logits: Tensor  # (52,)
    value: Tensor  # scalar


class HeartsNet(nn.Module):
    """Card-token Transformer with dual policy (52) and value heads."""

    def __init__(
        self,
        d_seat: int = 8,
        d_trick: int = 8,
        d_card: int = 16,
        d_suit: int = 4,
        d_role: int = 4,
        d_position: int = 4,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seat_emb = nn.Embedding(NUM_PLAYERS + 1, d_seat)
        self.trick_emb = nn.Embedding(CARDS_PER_HAND + 1, d_trick)
        self.card_id_emb = nn.Embedding(DECK_SIZE, d_card)
        self.suit_emb = nn.Embedding(5, d_suit)
        self.role_emb = nn.Embedding(NUM_PLAYERS, d_role)
        self.position_emb = nn.Embedding(NUM_PLAYERS + 1, d_position)

        card_in = 2 + d_seat + d_trick + d_card + d_position + 1
        global_in = (
            CARDS_PER_HAND * d_seat
            + CARDS_PER_HAND * d_suit
            + d_suit
            + d_role
            + 1
            + NUM_PLAYERS
            + d_trick
        )
        self.card_proj = nn.Linear(card_in, d_model)
        self.global_proj = nn.Linear(global_in, d_model)
        self.global_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.global_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.policy_head = nn.Linear(d_model, 1)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        obs: EncodedObservation | dict[str, Tensor],
        legal_mask: Tensor | None = None,
    ) -> PolicyValueOutput:
        tensors = self._as_tensors(obs)
        device = self.card_id_emb.weight.device
        tensors = {k: v.to(device) for k, v in tensors.items()}

        seat_ids = _map_none(tensors["played_by"], SEAT_NONE)
        trick_ids = _map_none(tensors["played_on_trick"], TRICK_NONE)
        position_ids = _map_none(tensors["play_position"], POSITION_NONE)
        card_ids = torch.arange(DECK_SIZE, device=device)

        x_card = torch.cat(
            [
                tensors["in_hand"].unsqueeze(-1),
                tensors["played"].unsqueeze(-1),
                self.seat_emb(seat_ids),
                self.trick_emb(trick_ids),
                self.card_id_emb(card_ids),
                self.position_emb(position_ids),
                tensors["broke_suit"].unsqueeze(-1),
            ],
            dim=-1,
        )
        card_tokens = self.card_proj(x_card)  # (52, d_model)

        leader_ids = _map_none(tensors["trick_leaders"], SEAT_NONE)
        trick_led_ids = _map_none(tensors["trick_led_suits"], SUIT_NONE)
        suit_id = _map_none(tensors["suit_led"].view(()), SUIT_NONE)
        role_id = tensors["my_role"].view(())
        g = torch.cat(
            [
                self.seat_emb(leader_ids).reshape(-1),
                self.suit_emb(trick_led_ids).reshape(-1),
                self.suit_emb(suit_id),
                self.role_emb(role_id),
                tensors["hearts_broken"].view(1),
                tensors["points_taken"],
                self.trick_emb(tensors["current_trick"].view(())),
            ],
            dim=0,
        )
        global_tok = self.global_proj(g) + self.global_token  # (d_model,)

        seq = torch.cat([global_tok.unsqueeze(0), card_tokens], dim=0).unsqueeze(0)
        encoded = self.encoder(seq).squeeze(0)  # (53, d_model)
        global_h = encoded[0]
        card_h = encoded[1:]

        logits = self.policy_head(card_h).squeeze(-1)
        value = self.value_head(global_h).squeeze(-1)

        if legal_mask is not None:
            mask = legal_mask.to(device=device, dtype=torch.bool)
            logits = logits.masked_fill(~mask, float("-inf"))
        return PolicyValueOutput(logits=logits, value=value)

    @staticmethod
    def _as_tensors(obs: EncodedObservation | dict[str, Tensor]) -> dict[str, Tensor]:
        if isinstance(obs, dict):
            return obs
        return observation_to_tensors(obs)


def observation_to_tensors(obs: EncodedObservation) -> dict[str, Tensor]:
    """Convert an EncodedObservation into a dict of tensors for the network / corpus."""
    return {
        "in_hand": torch.as_tensor(obs.in_hand, dtype=torch.float32),
        "played": torch.as_tensor(obs.played, dtype=torch.float32),
        "played_by": torch.as_tensor(obs.played_by, dtype=torch.long),
        "played_on_trick": torch.as_tensor(obs.played_on_trick, dtype=torch.long),
        "play_position": torch.as_tensor(obs.play_position, dtype=torch.long),
        "broke_suit": torch.as_tensor(obs.broke_suit, dtype=torch.float32),
        "trick_leaders": torch.as_tensor(obs.trick_leaders, dtype=torch.long),
        "trick_led_suits": torch.as_tensor(obs.trick_led_suits, dtype=torch.long),
        "suit_led": torch.as_tensor(obs.suit_led, dtype=torch.long),
        "my_role": torch.as_tensor(obs.my_role, dtype=torch.long),
        "hearts_broken": torch.as_tensor(obs.hearts_broken, dtype=torch.float32),
        "points_taken": torch.as_tensor(obs.points_taken, dtype=torch.float32),
        "current_trick": torch.as_tensor(obs.current_trick, dtype=torch.long),
    }


# Backward-compatible alias used by older imports/tests.
HeartsPolicyStem = HeartsNet


def legal_mask_from_cards(legal_cards: list) -> Tensor:
    """Build a length-52 bool mask from a list of ``Card`` objects."""
    mask = torch.zeros(DECK_SIZE, dtype=torch.bool)
    for card in legal_cards:
        mask[card.card_id] = True
    return mask
