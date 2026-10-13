from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from hearts.state import GameState

DEFAULT_DIFFICULTY = "random"
OMNISCIENT_DIFFICULTY = "omniscient"
DIFFICULTIES = ("random", "human", OMNISCIENT_DIFFICULTY)


@dataclass
class GameStore:
    games: dict[str, GameState] = field(default_factory=dict)
    difficulties: dict[str, str] = field(default_factory=dict)
    # Per-game review log. ``opening`` is the table at the first lead;
    # ``plays`` is one frame per card, in order.
    openings: dict[str, dict] = field(default_factory=dict)
    plays: dict[str, list] = field(default_factory=dict)
    human_player: int = 0

    def create(self, state: GameState, difficulty: str = DEFAULT_DIFFICULTY) -> str:
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"Unknown difficulty: {difficulty}")
        game_id = str(uuid.uuid4())
        self.games[game_id] = state
        self.difficulties[game_id] = difficulty
        return game_id

    def get(self, game_id: str) -> GameState | None:
        return self.games.get(game_id)

    def difficulty(self, game_id: str) -> str:
        return self.difficulties.get(game_id, DEFAULT_DIFFICULTY)

    def set(self, game_id: str, state: GameState) -> None:
        self.games[game_id] = state

    def delete(self, game_id: str) -> None:
        self.games.pop(game_id, None)
        self.difficulties.pop(game_id, None)
        self.openings.pop(game_id, None)
        self.plays.pop(game_id, None)

    def reset_replay(self, game_id: str) -> None:
        self.openings.pop(game_id, None)
        self.plays[game_id] = []


store = GameStore()
