from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from hearts.cards import Card, Deck, Rank, Suit
from hearts.config import GameConfig
from hearts.encode import POINTS_PER_ROUND, EncodedObservation, encode_state
from hearts.model import legal_mask_from_cards
from hearts.rules import CARDS_PER_HAND, NUM_PLAYERS, PassDirection, two_of_clubs_holder
from hearts.state import GameState, Phase, _blank_card_play_tricks, _blank_trick_leaders


def card_from_id(card_id: int) -> Card:
    """Rebuild a Card from its canonical 0–51 id."""
    suit = list(Suit)[card_id // 13]
    rank = list(Rank)[card_id % 13]
    return Card(rank, suit)


def normalize_round_return(points: int) -> float:
    """Map round points to z in [-1, 1] (lower points => higher return)."""
    z = 1.0 - 2.0 * (points / POINTS_PER_ROUND)
    return float(max(-1.0, min(1.0, z)))


class HoldRoundEnv:
    """Self-play environment for a single hold round (no passing)."""

    def __init__(self, config: GameConfig | None = None) -> None:
        self.config = config or GameConfig()
        self.state: GameState | None = None

    def reset(self, seed: int | None = None) -> GameState:
        cfg = self.config
        if seed is not None:
            cfg = replace(cfg, seed=seed)
        deck = Deck.fresh(cfg.seed)
        hands = deck.deal(NUM_PLAYERS, CARDS_PER_HAND)
        leader = two_of_clubs_holder(hands)
        leaders = _blank_trick_leaders()
        leaders[0] = leader
        self.state = GameState(
            config=cfg,
            phase=Phase.PLAYING,
            round_number=1,
            hands=hands,
            pass_direction=PassDirection.HOLD,
            pending_passes={},
            trick_number=1,
            trick_leader=leader,
            current_player=leader,
            current_trick=[],
            hearts_broken=False,
            tricks_won=[[] for _ in range(NUM_PLAYERS)],
            scores=[0] * NUM_PLAYERS,
            last_round_scores=None,
            history=[{"event": "deal", "round": 1}],
            card_play_tricks=_blank_card_play_tricks(),
            trick_leaders=leaders,
        )
        return self.state

    def step(self, card: Card) -> GameState:
        if self.state is None:
            raise RuntimeError("Call reset() before step()")
        player = self.state.current_player
        self.state = self.state.play_card(player, card)
        return self.state

    def observation(self, player: int | None = None) -> EncodedObservation:
        if self.state is None:
            raise RuntimeError("Call reset() before observation()")
        acting = self.state.current_player if player is None else player
        return encode_state(self.state, acting)

    def legal_mask(self, player: int | None = None) -> Tensor:
        if self.state is None:
            raise RuntimeError("Call reset() before legal_mask()")
        acting = self.state.current_player if player is None else player
        return legal_mask_from_cards(self.state.legal_moves_for(acting))

    @property
    def is_terminal(self) -> bool:
        return self.state is not None and self.state.phase in (Phase.ROUND_OVER, Phase.GAME_OVER)

    @property
    def round_scores(self) -> list[int] | None:
        if self.state is None:
            return None
        return self.state.last_round_scores

    def returns_by_player(self) -> list[float]:
        """Normalized returns z for each absolute seat after the round ends."""
        scores = self.round_scores
        if scores is None:
            raise RuntimeError("Round is not finished")
        return [normalize_round_return(p) for p in scores]
