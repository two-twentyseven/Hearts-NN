from __future__ import annotations

import random

import torch

from hearts import Card, GameConfig, GameState, PassDirection, Phase, Rank, Suit
from hearts.model import HeartsNet
from hearts.pimc import pimc_select, remaining_unknown_cards, sample_opponent_hands
from hearts.rules import NUM_PLAYERS
from hearts.state import DECK_SIZE


def c(rank: Rank, suit: Suit) -> Card:
    return Card(rank, suit)


def _tiny_mid_state() -> GameState:
    """Mid-round state with small hands so PIMC rollouts stay cheap."""
    two_clubs = c(Rank.TWO, Suit.CLUBS)
    king_clubs = c(Rank.KING, Suit.CLUBS)
    three_clubs = c(Rank.THREE, Suit.CLUBS)
    four_clubs = c(Rank.FOUR, Suit.CLUBS)
    two_spades = c(Rank.TWO, Suit.SPADES)
    three_spades = c(Rank.THREE, Suit.SPADES)
    queen_hearts = c(Rank.QUEEN, Suit.HEARTS)
    four_spades = c(Rank.FOUR, Suit.SPADES)
    five_clubs = c(Rank.FIVE, Suit.CLUBS)
    six_clubs = c(Rank.SIX, Suit.CLUBS)
    two_hearts = c(Rank.TWO, Suit.HEARTS)
    three_hearts = c(Rank.THREE, Suit.HEARTS)
    four_hearts = c(Rank.FOUR, Suit.HEARTS)
    five_hearts = c(Rank.FIVE, Suit.HEARTS)
    seven_clubs = c(Rank.SEVEN, Suit.CLUBS)
    eight_clubs = c(Rank.EIGHT, Suit.CLUBS)

    state = GameState(
        config=GameConfig(),
        phase=Phase.PLAYING,
        round_number=1,
        hands=[
            [two_clubs, four_spades, five_clubs, two_hearts],
            [king_clubs, two_spades, six_clubs, three_hearts],
            [three_clubs, three_spades, seven_clubs, four_hearts],
            [four_clubs, queen_hearts, eight_clubs, five_hearts],
        ],
        pass_direction=PassDirection.HOLD,
        trick_number=1,
        trick_leader=0,
        current_player=0,
        hearts_broken=False,
        scores=[0, 0, 0, 0],
        tricks_won=[[] for _ in range(NUM_PLAYERS)],
    )
    state = state.play_card(0, two_clubs)
    state = state.play_card(1, king_clubs)
    state = state.play_card(2, three_clubs)
    state = state.play_card(3, four_clubs)
    state = state.play_card(1, two_spades)
    state = state.play_card(2, three_spades)
    state = state.play_card(3, queen_hearts)
    state = state.play_card(0, four_spades)
    state = state.play_card(0, five_clubs)
    state = state.play_card(1, six_clubs)
    return state


def test_sample_opponent_hands_partition() -> None:
    state = _tiny_mid_state()
    acting = state.current_player
    rng = random.Random(1)
    hands = sample_opponent_hands(state, acting, rng)
    assert len(hands) == NUM_PLAYERS
    assert len(hands[acting]) == len(state.hands[acting])
    all_cards = [c for hand in hands for c in hand]
    assert len(all_cards) == sum(len(h) for h in state.hands)
    assert len({c.card_id for c in all_cards}) == len(all_cards)
    need = sum(len(state.hands[p]) for p in range(NUM_PLAYERS) if p != acting)
    assert sum(len(hands[p]) for p in range(NUM_PLAYERS) if p != acting) == need
    unknown = remaining_unknown_cards(state, acting)
    assert len(unknown) >= need


def test_pimc_returns_soft_legal_policy() -> None:
    state = _tiny_mid_state()
    net = HeartsNet(d_model=32, n_layers=1, n_heads=4, d_ff=64)
    net.eval()
    result = pimc_select(
        state,
        net,
        k=1,
        rollout_temperature=0.0,
        action_temperature=0.5,
        epsilon=0.0,
        rng=random.Random(0),
    )
    legal = state.legal_moves_for()
    legal_ids = {c.card_id for c in legal}
    assert result.pi.shape == (DECK_SIZE,)
    assert torch.isfinite(result.pi[list(legal_ids)]).all()
    assert abs(float(result.pi.sum()) - 1.0) < 1e-5
    for card_id in range(DECK_SIZE):
        if card_id not in legal_ids:
            assert float(result.pi[card_id]) == 0.0
    assert result.action in legal
