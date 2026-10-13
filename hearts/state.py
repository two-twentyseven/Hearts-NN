from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Any

from hearts.cards import Card, Deck
from hearts.config import GameConfig
from hearts.rules import (
    CARDS_PER_HAND,
    NUM_PLAYERS,
    PassDirection,
    finalize_round_scores,
    game_is_over,
    legal_plays,
    pass_direction_for_round,
    pass_target,
    trick_has_heart,
    trick_winner,
    two_of_clubs_holder,
)

DECK_SIZE = 52


def _blank_card_play_tricks() -> list[list[float]]:
    return [[0.0] * DECK_SIZE for _ in range(NUM_PLAYERS)]


def _blank_trick_leaders() -> list[int | None]:
    return [None] * CARDS_PER_HAND


class Phase(Enum):
    PASSING = auto()
    PLAYING = auto()
    ROUND_OVER = auto()
    GAME_OVER = auto()


class IllegalMoveReason(Enum):
    WRONG_PHASE = auto()
    NOT_YOUR_TURN = auto()
    CARD_NOT_IN_HAND = auto()
    ILLEGAL_CARD = auto()
    INVALID_PASS = auto()
    PASS_ALREADY_SUBMITTED = auto()
    ROUND_NOT_OVER = auto()
    GAME_ALREADY_OVER = auto()


class IllegalMoveError(Exception):
    def __init__(self, reason: IllegalMoveReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass
class GameState:
    config: GameConfig
    phase: Phase
    round_number: int
    hands: list[list[Card]]
    pass_direction: PassDirection
    pending_passes: dict[int, list[Card]] = field(default_factory=dict)
    trick_number: int = 1
    trick_leader: int = 0
    current_player: int = 0
    current_trick: list[tuple[int, Card]] = field(default_factory=list)
    hearts_broken: bool = False
    tricks_won: list[list[Card]] = field(default_factory=list)
    scores: list[int] = field(default_factory=list)
    last_round_scores: list[int] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    card_play_tricks: list[list[float]] = field(default_factory=_blank_card_play_tricks)
    trick_leaders: list[int | None] = field(default_factory=_blank_trick_leaders)

    def __post_init__(self) -> None:
        if (
            self.phase is Phase.PLAYING
            and 1 <= self.trick_number <= CARDS_PER_HAND
            and self.trick_leaders[self.trick_number - 1] is None
        ):
            self.trick_leaders[self.trick_number - 1] = self.trick_leader

    @classmethod
    def new_game(cls, config: GameConfig | None = None) -> GameState:
        cfg = config or GameConfig()
        state = cls(
            config=cfg,
            phase=Phase.PASSING,
            round_number=0,
            hands=[[] for _ in range(NUM_PLAYERS)],
            pass_direction=PassDirection.LEFT,
            scores=[0] * NUM_PLAYERS,
            tricks_won=[[] for _ in range(NUM_PLAYERS)],
        )
        return state.deal_new_round()

    def deal_new_round(self) -> GameState:
        if self.phase is Phase.GAME_OVER:
            raise IllegalMoveError(
                IllegalMoveReason.GAME_ALREADY_OVER,
                "Cannot deal a new round after the game has ended",
            )
        if self.phase is not Phase.ROUND_OVER and self.round_number > 0:
            raise IllegalMoveError(
                IllegalMoveReason.ROUND_NOT_OVER,
                "Current round must finish before dealing the next round",
            )

        round_number = self.round_number + 1
        direction = pass_direction_for_round(round_number)
        deck_seed = (
            None
            if self.config.seed is None
            else self.config.seed + round_number
        )
        deck = Deck.fresh(deck_seed)
        hands = deck.deal(NUM_PLAYERS, CARDS_PER_HAND)

        if direction is PassDirection.HOLD:
            leader = two_of_clubs_holder(hands)
            leaders = _blank_trick_leaders()
            leaders[0] = leader
            return replace(
                self,
                phase=Phase.PLAYING,
                round_number=round_number,
                hands=hands,
                pass_direction=direction,
                pending_passes={},
                trick_number=1,
                trick_leader=leader,
                current_player=leader,
                current_trick=[],
                hearts_broken=False,
                tricks_won=[[] for _ in range(NUM_PLAYERS)],
                last_round_scores=None,
                history=self.history + [{"event": "deal", "round": round_number}],
                card_play_tricks=_blank_card_play_tricks(),
                trick_leaders=leaders,
            )

        return replace(
            self,
            phase=Phase.PASSING,
            round_number=round_number,
            hands=hands,
            pass_direction=direction,
            pending_passes={},
            trick_number=1,
            trick_leader=0,
            current_player=0,
            current_trick=[],
            hearts_broken=False,
            tricks_won=[[] for _ in range(NUM_PLAYERS)],
            last_round_scores=None,
            history=self.history + [{"event": "deal", "round": round_number}],
            card_play_tricks=_blank_card_play_tricks(),
            trick_leaders=_blank_trick_leaders(),
        )

    def submit_pass(self, player: int, cards: list[Card]) -> GameState:
        if self.phase is not Phase.PASSING:
            raise IllegalMoveError(
                IllegalMoveReason.WRONG_PHASE,
                "Cards can only be passed during the passing phase",
            )
        if player in self.pending_passes:
            raise IllegalMoveError(
                IllegalMoveReason.PASS_ALREADY_SUBMITTED,
                f"Player {player} has already submitted a pass",
            )
        if len(cards) != 3:
            raise IllegalMoveError(
                IllegalMoveReason.INVALID_PASS,
                "Exactly three cards must be passed",
            )

        hand = self.hands[player]
        hand_copy = list(hand)
        for card in cards:
            if card not in hand_copy:
                raise IllegalMoveError(
                    IllegalMoveReason.CARD_NOT_IN_HAND,
                    f"Player {player} does not hold {card}",
                )
            hand_copy.remove(card)

        new_hands = deepcopy(self.hands)
        new_hands[player] = hand_copy
        new_passes = dict(self.pending_passes)
        new_passes[player] = list(cards)

        state = replace(
            self,
            hands=new_hands,
            pending_passes=new_passes,
            history=self.history
            + [{"event": "pass_submitted", "player": player, "cards": list(cards)}],
        )
        if len(new_passes) == NUM_PLAYERS:
            return state.apply_passes()
        return state

    def apply_passes(self) -> GameState:
        if self.phase is not Phase.PASSING:
            raise IllegalMoveError(
                IllegalMoveReason.WRONG_PHASE,
                "Passes can only be applied during the passing phase",
            )
        if len(self.pending_passes) != NUM_PLAYERS:
            raise IllegalMoveError(
                IllegalMoveReason.INVALID_PASS,
                "All four players must submit passes before applying them",
            )

        new_hands = deepcopy(self.hands)
        for player, cards in self.pending_passes.items():
            target = pass_target(player, self.pass_direction)
            new_hands[target].extend(cards)

        leader = two_of_clubs_holder(new_hands)
        leaders = _blank_trick_leaders()
        leaders[0] = leader
        return replace(
            self,
            phase=Phase.PLAYING,
            hands=new_hands,
            pending_passes={},
            trick_leader=leader,
            current_player=leader,
            history=self.history + [{"event": "passes_applied"}],
            card_play_tricks=_blank_card_play_tricks(),
            trick_leaders=leaders,
        )

    def legal_moves_for(self, player: int | None = None) -> list[Card]:
        player = self.current_player if player is None else player
        if self.phase is not Phase.PLAYING or player != self.current_player:
            return []
        is_leading = not self.current_trick
        led_suit = self.current_trick[0][1].suit if self.current_trick else None
        is_first_trick = self.trick_number == 1
        return legal_plays(
            self.hands[player],
            is_leading=is_leading,
            led_suit=led_suit,
            hearts_broken=self.hearts_broken,
            is_first_trick=is_first_trick,
            config=self.config,
        )

    def play_card(self, player: int, card: Card) -> GameState:
        if self.phase is not Phase.PLAYING:
            raise IllegalMoveError(
                IllegalMoveReason.WRONG_PHASE,
                "Cards can only be played during the playing phase",
            )
        if player != self.current_player:
            raise IllegalMoveError(
                IllegalMoveReason.NOT_YOUR_TURN,
                f"It is player {self.current_player}'s turn, not player {player}'s",
            )
        if card not in self.hands[player]:
            raise IllegalMoveError(
                IllegalMoveReason.CARD_NOT_IN_HAND,
                f"Player {player} does not hold {card}",
            )
        if card not in self.legal_moves_for(player):
            raise IllegalMoveError(
                IllegalMoveReason.ILLEGAL_CARD,
                f"{card} is not a legal play in the current trick",
            )

        new_hands = deepcopy(self.hands)
        new_hands[player].remove(card)
        new_trick = list(self.current_trick) + [(player, card)]
        new_plays = [row[:] for row in self.card_play_tricks]
        new_plays[player][card.card_id] = self.trick_number / float(CARDS_PER_HAND)
        state = replace(
            self,
            hands=new_hands,
            current_trick=new_trick,
            history=self.history + [{"event": "play", "player": player, "card": card}],
            card_play_tricks=new_plays,
        )

        if len(new_trick) < NUM_PLAYERS:
            return replace(
                state,
                current_player=(player + 1) % NUM_PLAYERS,
            )
        return state._complete_trick()

    def _complete_trick(self) -> GameState:
        winner = trick_winner(self.current_trick)
        trick_cards = [card for _, card in self.current_trick]
        new_tricks_won = deepcopy(self.tricks_won)
        new_tricks_won[winner].extend(trick_cards)

        hearts_broken = self.hearts_broken or trick_has_heart(self.current_trick)
        next_trick_number = self.trick_number + 1
        new_leaders = list(self.trick_leaders)
        if next_trick_number <= CARDS_PER_HAND:
            new_leaders[next_trick_number - 1] = winner

        state = replace(
            self,
            current_trick=[],
            tricks_won=new_tricks_won,
            hearts_broken=hearts_broken,
            trick_number=next_trick_number,
            trick_leader=winner,
            current_player=winner,
            history=self.history + [{"event": "trick_won", "player": winner}],
            trick_leaders=new_leaders,
        )

        if next_trick_number > CARDS_PER_HAND:
            return state._complete_round()
        return state

    def _complete_round(self) -> GameState:
        round_scores = finalize_round_scores(self.tricks_won, self.config)
        new_scores = [
            self.scores[player] + round_scores[player] for player in range(NUM_PLAYERS)
        ]
        phase = (
            Phase.GAME_OVER
            if game_is_over(new_scores, self.config)
            else Phase.ROUND_OVER
        )
        return replace(
            self,
            phase=phase,
            scores=new_scores,
            last_round_scores=round_scores,
            history=self.history
            + [{"event": "round_complete", "round_scores": round_scores}],
        )
