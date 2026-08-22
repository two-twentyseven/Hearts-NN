from __future__ import annotations

import random

from hearts.rules import NUM_PLAYERS
from hearts.state import GameState, Phase


def random_pass_cards(hand: list, rng: random.Random | None = None) -> list:
    rng = rng or random.Random()
    return rng.sample(hand, 3)


def advance_bots(state: GameState, human: int = 0, rng: random.Random | None = None) -> GameState:
    """Play random legal moves for all non-human players until human must act."""
    rng = rng or random.Random()

    if state.phase is Phase.PASSING:
        for player in range(NUM_PLAYERS):
            if player == human:
                continue
            if player in state.pending_passes:
                continue
            cards = random_pass_cards(state.hands[player], rng)
            state = state.submit_pass(player, cards)
        return state

    if state.phase is Phase.PLAYING:
        safety = 0
        while state.phase is Phase.PLAYING and state.current_player != human:
            moves = state.legal_moves_for()
            if not moves:
                break
            card = rng.choice(moves)
            state = state.play_card(state.current_player, card)
            safety += 1
            if safety > 52:
                break
        return state

    return state


def advance_one_bot(
    state: GameState,
    human: int = 0,
    rng: random.Random | None = None,
) -> tuple[GameState, dict | None]:
    """Advance exactly one visible bot action."""
    rng = rng or random.Random()
    if state.phase is not Phase.PLAYING or state.current_player == human:
        return state, None

    player = state.current_player
    moves = state.legal_moves_for()
    if not moves:
        return state, None

    card = rng.choice(moves)
    trick_before = list(state.current_trick)
    next_state = state.play_card(player, card)
    action = {
        "type": "play",
        "player": player,
        "card": card,
        "trick_winner": (
            next_state.trick_leader
            if len(trick_before) + 1 == NUM_PLAYERS
            else None
        ),
        "trick_cards": trick_before + [(player, card)],
    }
    return next_state, action


def advance_until_human(state: GameState, human: int = 0, rng: random.Random | None = None) -> GameState:
    """Advance bots until the human must act or the phase requires user confirmation."""
    rng = rng or random.Random()

    if state.phase in (Phase.ROUND_OVER, Phase.GAME_OVER):
        return state

    if state.phase is Phase.PASSING:
        state = advance_bots(state, human, rng)
        return state

    if state.phase is Phase.PLAYING:
        return advance_bots(state, human, rng)

    return state
