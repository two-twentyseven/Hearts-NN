from __future__ import annotations

import numpy as np

from hearts.cards import Card, Rank, Suit
from hearts.perfect.encode import (
    GLOBAL_DIM,
    LOC_IN_HAND,
    LOC_ON_TRICK,
    LOC_TAKEN,
    NUM_LOCATIONS,
    encode_perfect,
    legal_mask_for,
    normalize_points,
    relative_seat,
    rotate_to_absolute,
    rotate_to_relative,
)
from hearts.perfect.game import card_from_id, new_hold_round
from hearts.rules import NUM_PLAYERS
from hearts.state import DECK_SIZE


def _play_n(state, n: int):
    for _ in range(n):
        legal = state.legal_moves_for(state.current_player)
        state = state.play_card(state.current_player, legal[0])
    return state


def test_card_from_id_round_trips() -> None:
    for i in range(DECK_SIZE):
        assert card_from_id(i).card_id == i


def test_fresh_deal_every_card_in_a_hand() -> None:
    state = new_hold_round(seed=7)
    obs = encode_perfect(state, state.current_player)
    assert obs.locations.shape == (DECK_SIZE,)
    assert obs.globals.shape == (GLOBAL_DIM,)
    kinds = obs.locations // NUM_PLAYERS
    assert (kinds == LOC_IN_HAND).all()
    counts = np.bincount(obs.locations, minlength=NUM_LOCATIONS)
    assert list(counts[:4]) == [13, 13, 13, 13]
    # Acting player is seat 0 and leads the first trick.
    assert obs.globals[8] == 1.0  # leader one-hot rel 0
    assert obs.globals[12] == 1.0  # my position 0
    assert obs.globals[16] == 0.0  # hearts not broken
    assert obs.globals[17] == 0.0  # trick 1


def test_mid_trick_locations_and_rotation() -> None:
    state = new_hold_round(seed=7)
    state = _play_n(state, 6)  # one full trick + 2 cards into trick 2
    acting = state.current_player
    obs = encode_perfect(state, acting)
    kinds = obs.locations // NUM_PLAYERS
    seats = obs.locations % NUM_PLAYERS
    assert (kinds == LOC_ON_TRICK).sum() == 2
    assert (kinds == LOC_TAKEN).sum() == 4
    assert (kinds == LOC_IN_HAND).sum() == 46
    # The on-trick cards belong to the two seats before the actor: rel 3 (leader) and... the leader led.
    for seat, card in state.current_trick:
        assert obs.locations[card.card_id] == LOC_ON_TRICK * 4 + relative_seat(seat, acting)
    winner = state.trick_leader
    for card in state.tricks_won[winner]:
        assert seats[card.card_id] == relative_seat(winner, acting)
    for seat in range(NUM_PLAYERS):
        for card in state.hands[seat]:
            assert obs.locations[card.card_id] == LOC_IN_HAND * 4 + relative_seat(seat, acting)
    # Encoding from another seat is a pure seat rotation of the same information.
    other = (acting + 1) % NUM_PLAYERS
    obs_other = encode_perfect(state, other)
    for cid in range(DECK_SIZE):
        kind, rel = divmod(int(obs.locations[cid]), NUM_PLAYERS)
        kind_o, rel_o = divmod(int(obs_other.locations[cid]), NUM_PLAYERS)
        assert kind == kind_o
        assert (rel - rel_o) % NUM_PLAYERS == 1
    assert obs.globals[17] > 0.0


def test_round_points_globals_track_tricks_won() -> None:
    state = new_hold_round(seed=11)
    # Play until someone has taken a point card.
    for _ in range(52):
        if any(c.is_point_card(state.config) for won in state.tricks_won for c in won):
            break
        legal = state.legal_moves_for(state.current_player)
        state = state.play_card(state.current_player, legal[-1])
    acting = state.current_player
    obs = encode_perfect(state, acting)
    for seat in range(NUM_PLAYERS):
        pts = sum(c.points(state.config) for c in state.tricks_won[seat])
        assert np.isclose(obs.globals[relative_seat(seat, acting)], pts / 26.0)


def test_rotate_helpers_round_trip() -> None:
    v = np.array([10.0, 20.0, 30.0, 40.0])
    for acting in range(NUM_PLAYERS):
        rel = rotate_to_relative(v, acting)
        assert rel[0] == v[acting]
        assert rel[1] == v[(acting + 1) % 4]
        assert np.array_equal(rotate_to_absolute(rel, acting), v)


def test_normalize_points() -> None:
    assert normalize_points(0) == 1.0
    assert normalize_points(26) == -1.0
    assert np.isclose(normalize_points(13), 0.0)


def test_legal_mask_matches_engine() -> None:
    state = new_hold_round(seed=3)
    mask = legal_mask_for(state)
    assert mask.sum() == 1
    assert card_from_id(int(np.flatnonzero(mask)[0])) == Card(Rank.TWO, Suit.CLUBS)
