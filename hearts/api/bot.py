from __future__ import annotations

import random

from hearts.api.game_store import DEFAULT_DIFFICULTY, OMNISCIENT_DIFFICULTY
from hearts.encode import encode_state
from hearts.human_bot import select_card, select_pass
from hearts.model import legal_mask_from_cards
from hearts.perfect.agent import shared_bot
from hearts.rules import NUM_PLAYERS
from hearts.state import GameState, Phase


def random_pass_cards(hand: list, rng: random.Random | None = None) -> list:
    rng = rng or random.Random()
    return rng.sample(hand, 3)


def random_play(state: GameState, rng: random.Random | None = None):
    rng = rng or random.Random()
    legal = state.legal_moves_for()
    if not legal:
        return None
    return rng.choice(legal)


def heuristic_play(state: GameState):
    """Pick a card with the unlearned heuristic bot from the public observation."""
    legal = state.legal_moves_for()
    if not legal:
        return None
    obs = encode_state(state, state.current_player)
    return select_card(obs, legal_mask_from_cards(legal))


def omniscient_play(state: GameState):
    """Perfect-information play: trained PerfectNet + paranoid MCTS over the full state.

    Unlike the other bots this never looks at ``state.history``; it reads every
    hand directly, so it works on any server-side state regardless of how the
    history was (or was not) maintained.
    """
    return shared_bot().play(state)


def bot_pass_cards(
    hand: list,
    difficulty: str = DEFAULT_DIFFICULTY,
    rng: random.Random | None = None,
) -> list:
    if difficulty == "human":
        return select_pass(hand)
    if difficulty == OMNISCIENT_DIFFICULTY:
        return shared_bot().pass_cards(hand)
    return random_pass_cards(hand, rng)


def bot_play(
    state: GameState,
    difficulty: str = DEFAULT_DIFFICULTY,
    rng: random.Random | None = None,
):
    if difficulty == "human":
        return heuristic_play(state)
    if difficulty == OMNISCIENT_DIFFICULTY:
        return omniscient_play(state)
    return random_play(state, rng)


def advance_bots(
    state: GameState,
    human: int = 0,
    rng: random.Random | None = None,
    difficulty: str = DEFAULT_DIFFICULTY,
) -> GameState:
    """Advance non-human seats until the human must act."""
    rng = rng or random.Random()

    if state.phase is Phase.PASSING:
        for player in range(NUM_PLAYERS):
            if player == human:
                continue
            if player in state.pending_passes:
                continue
            cards = bot_pass_cards(state.hands[player], difficulty, rng)
            state = state.submit_pass(player, cards)
        return state

    if state.phase is Phase.PLAYING:
        safety = 0
        while state.phase is Phase.PLAYING and state.current_player != human:
            card = bot_play(state, difficulty, rng)
            if card is None:
                break
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
    difficulty: str = DEFAULT_DIFFICULTY,
) -> tuple[GameState, dict | None]:
    """Advance exactly one visible bot action."""
    rng = rng or random.Random()
    if state.phase is not Phase.PLAYING or state.current_player == human:
        return state, None

    player = state.current_player
    card = bot_play(state, difficulty, rng)
    if card is None:
        return state, None

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


def advance_until_human(
    state: GameState,
    human: int = 0,
    rng: random.Random | None = None,
    difficulty: str = DEFAULT_DIFFICULTY,
) -> GameState:
    """Advance bots until the human must act or the phase requires user confirmation."""
    rng = rng or random.Random()

    if state.phase in (Phase.ROUND_OVER, Phase.GAME_OVER):
        return state

    if state.phase is Phase.PASSING:
        state = advance_bots(state, human, rng, difficulty)
        return state

    if state.phase is Phase.PLAYING:
        return advance_bots(state, human, rng, difficulty)

    return state
