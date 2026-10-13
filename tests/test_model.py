from __future__ import annotations

import torch

from hearts import Card, GameConfig, GameState, PassDirection, Phase, Rank, Suit, encode_state
from hearts.model import HeartsNet, legal_mask_from_cards
from hearts.rules import NUM_PLAYERS
from hearts.state import DECK_SIZE


def c(rank: Rank, suit: Suit) -> Card:
    return Card(rank, suit)


def _mid_round_state() -> GameState:
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
        pass_direction=PassDirection.LEFT,
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


def test_hearts_net_policy_value_and_mask() -> None:
    state = _mid_round_state()
    acting = state.current_player
    obs = encode_state(state, acting)
    legal = state.legal_moves_for(acting)
    mask = legal_mask_from_cards(legal)

    model = HeartsNet(d_model=64, n_layers=2, n_heads=4, d_ff=128)
    model.eval()
    with torch.no_grad():
        out = model(obs, legal_mask=mask)

    assert out.logits.shape == (DECK_SIZE,)
    assert out.value.ndim == 0
    assert torch.isfinite(out.logits[mask]).all()
    assert torch.isneginf(out.logits[~mask]).all()
    assert -1.0 <= float(out.value) <= 1.0

    unmasked = model(obs, legal_mask=None)
    assert torch.isfinite(unmasked.logits).all()
