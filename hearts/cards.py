from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, auto

from hearts.config import GameConfig


class Suit(Enum):
    CLUBS = auto()
    DIAMONDS = auto()
    HEARTS = auto()
    SPADES = auto()


class Rank(Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


@dataclass(frozen=True, order=True)
class Card:
    rank: Rank
    suit: Suit

    @property
    def is_heart(self) -> bool:
        return self.suit is Suit.HEARTS

    @property
    def is_two_of_clubs(self) -> bool:
        return self.suit is Suit.CLUBS and self.rank is Rank.TWO

    @property
    def is_queen_of_spades(self) -> bool:
        return self.suit is Suit.SPADES and self.rank is Rank.QUEEN

    @property
    def is_jack_of_diamonds(self) -> bool:
        return self.suit is Suit.DIAMONDS and self.rank is Rank.JACK

    def is_point_card(self, config: GameConfig) -> bool:
        if self.is_heart or self.is_queen_of_spades:
            return True
        return config.use_jack_of_diamonds and self.is_jack_of_diamonds

    def points(self, config: GameConfig) -> int:
        if self.is_heart:
            return 1
        if self.is_queen_of_spades:
            return 13
        if config.use_jack_of_diamonds and self.is_jack_of_diamonds:
            return -10
        return 0


class Deck:
    """Standard 52-card deck."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._cards = [
            Card(rank, suit)
            for suit in Suit
            for rank in Rank
        ]

    def shuffle(self) -> None:
        self._rng.shuffle(self._cards)

    def deal(self, num_players: int = 4, hand_size: int = 13) -> list[list[Card]]:
        if len(self._cards) != num_players * hand_size:
            raise ValueError("Deck must contain exactly num_players * hand_size cards")
        hands: list[list[Card]] = []
        for player in range(num_players):
            start = player * hand_size
            end = start + hand_size
            hands.append(list(self._cards[start:end]))
        return hands

    @classmethod
    def fresh(cls, seed: int | None = None) -> Deck:
        deck = cls(seed)
        deck.shuffle()
        return deck
