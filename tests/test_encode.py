from __future__ import annotations

import numpy as np

from hearts import Card, GameConfig, GameState, PassDirection, Phase, Rank, Suit, encode_state
from hearts.encode import NONE_INDEX, EncodedObservation
from hearts.rules import CARDS_PER_HAND, NUM_PLAYERS
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

    # Trick 1: 2C-KC-3C-4C, player 1 wins (and leads trick 2).
    state = state.play_card(0, two_clubs)
    state = state.play_card(1, king_clubs)
    state = state.play_card(2, three_clubs)
    state = state.play_card(3, four_clubs)
    # Trick 2: 2S-3S-QH-4S, player 0 wins; hearts broken; 1 point to player 0.
    state = state.play_card(1, two_spades)
    state = state.play_card(2, three_spades)
    state = state.play_card(3, queen_hearts)
    state = state.play_card(0, four_spades)
    # Trick 3 in progress: player 0 leads 5C, player 1 follows 6C.
    state = state.play_card(0, five_clubs)
    state = state.play_card(1, six_clubs)
    return state


def test_card_id_is_suit_times_13_plus_rank() -> None:
    two_clubs = c(Rank.TWO, Suit.CLUBS)
    ace_hearts = c(Rank.ACE, Suit.HEARTS)
    ace_spades = c(Rank.ACE, Suit.SPADES)
    assert two_clubs.card_id == 0
    assert ace_hearts.card_id == 2 * 13 + 12
    assert ace_spades.card_id == 51


def test_encode_state_mid_round() -> None:
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

    state = _mid_round_state()
    acting = 0
    obs = encode_state(state, acting)
    assert isinstance(obs, EncodedObservation)

    remaining = state.hands[acting]
    assert int(obs.in_hand.sum()) == len(remaining)
    for card in remaining:
        assert obs.in_hand[card.card_id] == 1.0

    played = [
        (0, two_clubs, 0),
        (1, king_clubs, 0),
        (2, three_clubs, 0),
        (3, four_clubs, 0),
        (1, two_spades, 1),
        (2, three_spades, 1),
        (3, queen_hearts, 1),
        (0, four_spades, 1),
        (0, five_clubs, 2),
        (1, six_clubs, 2),
    ]
    assert int(obs.played.sum()) == len(played)
    for seat, card, trick_index in played:
        assert obs.played[card.card_id] == 1.0
        assert obs.played_by[card.card_id] == seat
        assert obs.played_on_trick[card.card_id] == trick_index

    assert obs.play_position[two_clubs.card_id] == 0
    assert obs.play_position[king_clubs.card_id] == 1
    assert obs.play_position[queen_hearts.card_id] == 2
    assert obs.broke_suit[queen_hearts.card_id] == 1.0  # void in spades on trick 2
    assert obs.broke_suit[two_clubs.card_id] == 0.0
    assert obs.broke_suit[four_spades.card_id] == 0.0

    np.testing.assert_array_equal(obs.trick_led_suits[:3], [0, 3, 0])  # clubs, spades, clubs
    assert obs.trick_led_suits[3] == NONE_INDEX

    unplayed = obs.played == 0.0
    assert np.all(obs.played_by[unplayed] == NONE_INDEX)
    assert np.all(obs.played_on_trick[unplayed] == NONE_INDEX)
    assert np.all(obs.play_position[unplayed] == NONE_INDEX)
    assert np.all(obs.broke_suit[unplayed] == 0.0)

    # Each played card has exactly one (seat, trick) assignment.
    for card_id in range(DECK_SIZE):
        if obs.played[card_id] == 1.0:
            assert 0 <= obs.played_by[card_id] < NUM_PLAYERS
            assert 0 <= obs.played_on_trick[card_id] < CARDS_PER_HAND
        else:
            assert obs.played_by[card_id] == NONE_INDEX
            assert obs.played_on_trick[card_id] == NONE_INDEX
            assert obs.play_position[card_id] == NONE_INDEX
            assert obs.broke_suit[card_id] == 0.0

    np.testing.assert_array_equal(
        obs.trick_leaders,
        [0, 1, 0, NONE_INDEX, NONE_INDEX, NONE_INDEX, NONE_INDEX,
         NONE_INDEX, NONE_INDEX, NONE_INDEX, NONE_INDEX, NONE_INDEX, NONE_INDEX],
    )
    assert obs.suit_led == 0  # clubs
    assert obs.my_role == 0  # leader
    assert obs.hearts_broken == 1.0
    np.testing.assert_allclose(obs.points_taken, [1.0 / 26.0, 0.0, 0.0, 0.0])
    assert obs.current_trick == 2

    from_p2 = encode_state(state, acting_player=2)
    np.testing.assert_array_equal(from_p2.trick_leaders[:3], [2, 3, 2])
    assert from_p2.my_role == 2  # third to play when leader is relative seat 2


def test_encode_state_empty_trick_has_no_suit_led() -> None:
    state = GameState.new_game(GameConfig(seed=0))
    # Hold round or after passes: find a playing state with empty current trick.
    if state.phase is Phase.PASSING:
        for player in range(NUM_PLAYERS):
            state = state.submit_pass(player, state.hands[player][:3])
    assert state.phase is Phase.PLAYING
    assert state.current_trick == []
    obs = encode_state(state, state.current_player)
    assert obs.suit_led == NONE_INDEX
    assert obs.current_trick == 0
    assert obs.my_role == 0
    assert np.all(obs.trick_led_suits == NONE_INDEX)
