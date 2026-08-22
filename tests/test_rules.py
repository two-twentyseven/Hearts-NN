from __future__ import annotations

import pytest

from hearts import (
    Card,
    GameConfig,
    GameState,
    IllegalMoveError,
    IllegalMoveReason,
    PassDirection,
    Phase,
    Rank,
    Suit,
)
from hearts.rules import (
    apply_shoot_the_moon,
    finalize_round_scores,
    legal_plays,
    pass_direction_for_round,
    shot_the_moon,
)


def c(rank: Rank, suit: Suit) -> Card:
    return Card(rank, suit)


def test_pass_direction_cycles() -> None:
    assert pass_direction_for_round(1) is PassDirection.LEFT
    assert pass_direction_for_round(2) is PassDirection.RIGHT
    assert pass_direction_for_round(3) is PassDirection.ACROSS
    assert pass_direction_for_round(4) is PassDirection.HOLD
    assert pass_direction_for_round(5) is PassDirection.LEFT


def test_hold_round_skips_passing() -> None:
    state = GameState.new_game(GameConfig(seed=42))
    while state.round_number < 4:
        state = _finish_passing_and_round(state)
    assert state.round_number == 4
    assert state.pass_direction is PassDirection.HOLD
    assert state.phase is Phase.PLAYING


def test_two_of_clubs_must_lead_first_trick() -> None:
    two_clubs = c(Rank.TWO, Suit.CLUBS)
    three_clubs = c(Rank.THREE, Suit.CLUBS)
    queen_spades = c(Rank.QUEEN, Suit.SPADES)
    ace_hearts = c(Rank.ACE, Suit.HEARTS)

    state = _playing_state(
        hands=[
            [two_clubs, three_clubs, queen_spades],
            [c(Rank.FOUR, Suit.CLUBS)],
            [c(Rank.FIVE, Suit.CLUBS)],
            [c(Rank.SIX, Suit.CLUBS)],
        ],
        leader=0,
    )

    assert state.legal_moves_for(0) == [two_clubs]

    with pytest.raises(IllegalMoveError) as exc:
        state.play_card(0, three_clubs)
    assert exc.value.reason is IllegalMoveReason.ILLEGAL_CARD

    state = state.play_card(0, two_clubs)
    assert state.current_player == 1


def test_no_point_cards_on_first_trick_unless_forced() -> None:
    config = GameConfig()
    queen_spades = c(Rank.QUEEN, Suit.SPADES)
    ace_hearts = c(Rank.ACE, Suit.HEARTS)
    four_spades = c(Rank.FOUR, Suit.SPADES)

    legal = legal_plays(
        [queen_spades, ace_hearts, four_spades],
        is_leading=False,
        led_suit=Suit.CLUBS,
        hearts_broken=False,
        is_first_trick=True,
        config=config,
    )
    assert legal == [four_spades]

    forced = legal_plays(
        [queen_spades, ace_hearts],
        is_leading=False,
        led_suit=Suit.CLUBS,
        hearts_broken=False,
        is_first_trick=True,
        config=config,
    )
    assert set(forced) == {queen_spades, ace_hearts}


def test_follow_suit_required() -> None:
    config = GameConfig()
    five_hearts = c(Rank.FIVE, Suit.HEARTS)
    seven_clubs = c(Rank.SEVEN, Suit.CLUBS)

    legal = legal_plays(
        [five_hearts, seven_clubs],
        is_leading=False,
        led_suit=Suit.CLUBS,
        hearts_broken=True,
        is_first_trick=False,
        config=config,
    )
    assert legal == [seven_clubs]


def test_hearts_lead_blocked_until_broken() -> None:
    config = GameConfig()
    five_hearts = c(Rank.FIVE, Suit.HEARTS)
    seven_clubs = c(Rank.SEVEN, Suit.CLUBS)

    blocked = legal_plays(
        [five_hearts, seven_clubs],
        is_leading=True,
        led_suit=None,
        hearts_broken=False,
        is_first_trick=False,
        config=config,
    )
    assert blocked == [seven_clubs]

    allowed = legal_plays(
        [five_hearts],
        is_leading=True,
        led_suit=None,
        hearts_broken=False,
        is_first_trick=False,
        config=config,
    )
    assert allowed == [five_hearts]


def test_hearts_broken_when_heart_is_played() -> None:
    state = _playing_state(
        hands=[
            [c(Rank.TWO, Suit.CLUBS)],
            [c(Rank.THREE, Suit.CLUBS)],
            [c(Rank.FOUR, Suit.CLUBS)],
            [c(Rank.FIVE, Suit.HEARTS)],
        ],
        leader=0,
        trick_number=2,
        hearts_broken=False,
    )
    state = state.play_card(0, c(Rank.TWO, Suit.CLUBS))
    state = state.play_card(1, c(Rank.THREE, Suit.CLUBS))
    state = state.play_card(2, c(Rank.FOUR, Suit.CLUBS))
    state = state.play_card(3, c(Rank.FIVE, Suit.HEARTS))
    assert state.hearts_broken is True


def test_jack_of_diamonds_scores_negative_ten() -> None:
    config = GameConfig(use_jack_of_diamonds=True)
    jack = c(Rank.JACK, Suit.DIAMONDS)
    queen = c(Rank.QUEEN, Suit.SPADES)
    ace_hearts = c(Rank.ACE, Suit.HEARTS)

    tricks_won = [[jack], [queen, ace_hearts], [], []]
    scores = finalize_round_scores(tricks_won, config)
    assert scores == [-10, 14, 0, 0]


def test_jack_disabled_has_no_point_value() -> None:
    config = GameConfig(use_jack_of_diamonds=False)
    jack = c(Rank.JACK, Suit.DIAMONDS)
    assert jack.points(config) == 0


def test_shoot_the_moon() -> None:
    hearts = [Card(rank, Suit.HEARTS) for rank in Rank]
    queen = c(Rank.QUEEN, Suit.SPADES)
    tricks_won = [hearts + [queen], [], [], []]

    assert shot_the_moon(tricks_won) == 0
    adjusted = apply_shoot_the_moon([26, 3, 5, 1], shooter=0)
    assert adjusted == [0, 26, 26, 26]


def test_game_ends_at_target_score() -> None:
    config = GameConfig(target_score=100, seed=1)
    state = GameState.new_game(config)
    state = GameState(
        config=config,
        phase=Phase.ROUND_OVER,
        round_number=1,
        hands=[[] for _ in range(4)],
        pass_direction=PassDirection.LEFT,
        tricks_won=[[], [], [], []],
        scores=[97, 20, 15, 10],
    )
    state = state.deal_new_round()
    state = _force_round_end(state, round_scores=[5, 0, 0, 0])
    assert state.scores[0] == 102
    assert state.phase is Phase.GAME_OVER


def test_pass_exchange_left() -> None:
    pass_sets = [
        [c(Rank.ACE, Suit.CLUBS), c(Rank.KING, Suit.CLUBS), c(Rank.QUEEN, Suit.CLUBS)],
        [c(Rank.JACK, Suit.CLUBS), c(Rank.TEN, Suit.CLUBS), c(Rank.NINE, Suit.CLUBS)],
        [c(Rank.EIGHT, Suit.CLUBS), c(Rank.SEVEN, Suit.CLUBS), c(Rank.SIX, Suit.CLUBS)],
        [c(Rank.FIVE, Suit.CLUBS), c(Rank.FOUR, Suit.CLUBS), c(Rank.THREE, Suit.CLUBS)],
    ]

    state = GameState(
        config=GameConfig(),
        phase=Phase.PASSING,
        round_number=1,
        hands=[
            pass_sets[0] + [c(Rank.TWO, Suit.CLUBS)],
            pass_sets[1] + [c(Rank.TWO, Suit.DIAMONDS)],
            pass_sets[2] + [c(Rank.TWO, Suit.HEARTS)],
            pass_sets[3] + [c(Rank.TWO, Suit.SPADES)],
        ],
        pass_direction=PassDirection.LEFT,
        scores=[0, 0, 0, 0],
        tricks_won=[[] for _ in range(4)],
    )

    for player in range(4):
        state = state.submit_pass(player, pass_sets[player])

    assert state.phase is Phase.PLAYING
    assert pass_sets[0][0] in state.hands[1]
    assert pass_sets[0][0] not in state.hands[0]


def _playing_state(
    hands: list[list[Card]],
    leader: int,
    *,
    trick_number: int = 1,
    hearts_broken: bool = False,
    config: GameConfig | None = None,
) -> GameState:
    return GameState(
        config=config or GameConfig(),
        phase=Phase.PLAYING,
        round_number=1,
        hands=hands,
        pass_direction=PassDirection.LEFT,
        trick_number=trick_number,
        trick_leader=leader,
        current_player=leader,
        hearts_broken=hearts_broken,
        scores=[0, 0, 0, 0],
        tricks_won=[[] for _ in range(4)],
    )


def _finish_passing_and_round(state: GameState) -> GameState:
    if state.phase is Phase.PASSING:
        for player in range(4):
            cards = state.hands[player][:3]
            state = state.submit_pass(player, cards)
    state = _play_out_round(state)
    if state.phase is Phase.ROUND_OVER:
        state = state.deal_new_round()
    return state


def _play_out_round(state: GameState) -> GameState:
    while state.phase is Phase.PLAYING:
        moves = state.legal_moves_for()
        assert moves, "Expected at least one legal move"
        state = state.play_card(state.current_player, moves[0])
    return state


def _force_round_end(state: GameState, round_scores: list[int]) -> GameState:
    new_scores = [
        state.scores[player] + round_scores[player] for player in range(4)
    ]
    phase = Phase.GAME_OVER if max(new_scores) >= state.config.target_score else Phase.ROUND_OVER
    return GameState(
        config=state.config,
        phase=phase,
        round_number=state.round_number,
        hands=state.hands,
        pass_direction=state.pass_direction,
        tricks_won=state.tricks_won,
        scores=new_scores,
        last_round_scores=round_scores,
    )
