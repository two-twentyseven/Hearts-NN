from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from hearts.state import GameState


@dataclass
class GameStore:
    games: dict[str, GameState] = field(default_factory=dict)
    human_player: int = 0

    def create(self, state: GameState) -> str:
        game_id = str(uuid.uuid4())
        self.games[game_id] = state
        return game_id

    def get(self, game_id: str) -> GameState | None:
        return self.games.get(game_id)

    def set(self, game_id: str, state: GameState) -> None:
        self.games[game_id] = state

    def delete(self, game_id: str) -> None:
        self.games.pop(game_id, None)


store = GameStore()
