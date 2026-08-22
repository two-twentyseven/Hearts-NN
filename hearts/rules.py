from __future__ import annotations

from enum import Enum, auto

from hearts.cards import Card, Rank, Suit
from hearts.config import GameConfig

NUM_PLAYERS = 4
CARDS_PER_HAND = 13
MOON_PENALTY = 26


class PassDirection(Enum):
    LEFT = auto()
    RIGHT = auto()
    ACROSS = auto()
    HOLD = auto()


def pass_direction_for_round(round_number: int) -> PassDirection:
    """Round 1=left, 2=right, 3=across, 4=hold, then repeat."""
    return list(PassDirection)[(round_number - 1) % 4]


def pass_target(player: int, direction: PassDirection) -> int:
    if direction is PassDirection.LEFT:
        return (player + 1) % NUM_PLAYERS
    if direction is PassDirection.RIGHT:
        return (player - 1) % NUM_PLAYERS
    if direction is PassDirection.ACROSS:
        return (player + 2) % NUM_PLAYERS
    raise ValueError("HOLD pass direction has no target")


def two_of_clubs_holder(hands: list[list[Card]]) -> int:
    for player, hand in enumerate(hands):
        if any(card.is_two_of_clubs for card in hand):
            return player
    raise ValueError("No player holds the 2 of clubs")


def _cards_of_suit(hand: list[Card], suit: Suit) -> list[Card]:
    return [card for card in hand if card.suit is suit]


def _hand_has_only_hearts(hand: list[Card]) -> bool:
    return bool(hand) and all(card.is_heart for card in hand)


def _is_forced_point_card_play(
    hand: list[Card],
    led_suit: Suit,
    config: GameConfig,
) -> bool:
    """True when every legal follow option is a point card."""
    followable = _cards_of_suit(hand, led_suit)
    if followable:
        return all(card.is_point_card(config) for card in followable)
    return all(card.is_point_card(config) for card in hand)


def legal_plays(
    hand: list[Card],
    *,
    is_leading: bool,
    led_suit: Suit | None,
    hearts_broken: bool,
    is_first_trick: bool,
    config: GameConfig,
) -> list[Card]:
    if not hand:
        return []

    candidates = list(hand)

    if is_leading:
        if not hearts_broken and not _hand_has_only_hearts(hand):
            candidates = [card for card in candidates if not card.is_heart]
        if is_first_trick:
            two_clubs = [card for card in candidates if card.is_two_of_clubs]
            if two_clubs:
                return two_clubs
            candidates = [
                card for card in candidates if not card.is_point_card(config)
            ]
        return candidates

    assert led_suit is not None
    followable = _cards_of_suit(hand, led_suit)
    if followable:
        candidates = followable
    if is_first_trick and not _is_forced_point_card_play(hand, led_suit, config):
        candidates = [
            card for card in candidates if not card.is_point_card(config)
        ]
    return candidates


def trick_winner(trick: list[tuple[int, Card]]) -> int:
    if not trick:
        raise ValueError("Cannot resolve an empty trick")
    led_suit = trick[0][1].suit
    winner_player, winner_card = trick[0]
    for player, card in trick[1:]:
        if card.suit is led_suit and card.rank.value > winner_card.rank.value:
            winner_player, winner_card = player, card
    return winner_player


def trick_has_heart(trick: list[tuple[int, Card]]) -> bool:
    return any(card.is_heart for _, card in trick)


def score_captured_cards(cards: list[Card], config: GameConfig) -> int:
    return sum(card.points(config) for card in cards)


def round_scores(tricks_won: list[list[Card]], config: GameConfig) -> list[int]:
    return [score_captured_cards(cards, config) for cards in tricks_won]


def shot_the_moon(tricks_won: list[list[Card]]) -> int | None:
    """Return the player who captured all 13 hearts and the queen of spades."""
    for player, cards in enumerate(tricks_won):
        hearts = sum(1 for card in cards if card.is_heart)
        has_queen = any(card.is_queen_of_spades for card in cards)
        if hearts == 13 and has_queen:
            return player
    return None


def apply_shoot_the_moon(round_scores: list[int], shooter: int) -> list[int]:
    adjusted = list(round_scores)
    adjusted[shooter] = 0
    for player in range(NUM_PLAYERS):
        if player != shooter:
            adjusted[player] = MOON_PENALTY
    return adjusted


def finalize_round_scores(
    tricks_won: list[list[Card]],
    config: GameConfig,
) -> list[int]:
    scores = round_scores(tricks_won, config)
    shooter = shot_the_moon(tricks_won)
    if shooter is not None:
        scores = apply_shoot_the_moon(scores, shooter)
    return scores


def game_is_over(scores: list[int], config: GameConfig) -> bool:
    return max(scores) >= config.target_score
