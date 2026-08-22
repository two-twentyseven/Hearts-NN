from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hearts.api.bot import advance_bots, advance_one_bot
from hearts.api.game_store import store
from hearts.api.serialize import card_to_id, cards_from_ids, card_from_id, state_for_viewer
from hearts.config import GameConfig
from hearts.rules import NUM_PLAYERS, pass_target
from hearts.state import GameState, IllegalMoveError

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="Hearts")

@app.middleware("http")
async def disable_static_cache(request: Request, call_next) -> Response:
    response = await call_next(request)
    if request.url.path.startswith(("/js/", "/css/", "/assets/")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PassRequest(BaseModel):
    cards: list[str]


class PlayRequest(BaseModel):
    card: str


def _view(game_id: str, state: GameState) -> dict:
    return state_for_viewer(state, store.human_player, game_id=game_id)


def _get_state(game_id: str) -> GameState:
    state = store.get(game_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Game not found")
    return state


def _serialize_action(action: dict | None) -> dict | None:
    if action is None:
        return None
    serialized = dict(action)
    if "card" in serialized:
        serialized["card"] = card_to_id(serialized["card"])
    if "cards" in serialized:
        serialized["cards"] = [card_to_id(card) for card in serialized["cards"]]
    if "incoming" in serialized:
        serialized["incoming"] = [
            card_to_id(card) for card in serialized["incoming"]
        ]
    if "trick_cards" in serialized:
        serialized["trick_cards"] = [
            {"player": player, "card": card_to_id(card)}
            for player, card in serialized["trick_cards"]
        ]
    return serialized


def _save_and_view(
    game_id: str,
    state: GameState,
    *,
    action: dict | None = None,
) -> dict:
    store.set(game_id, state)
    view = _view(game_id, state)
    view["action"] = _serialize_action(action)
    return view


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "engine": "hearts"}


@app.post("/api/games")
def create_game() -> dict:
    state = GameState.new_game(GameConfig())
    game_id = store.create(state)
    return _save_and_view(game_id, state)


@app.get("/api/games/{game_id}")
def get_game(game_id: str) -> dict:
    state = _get_state(game_id)
    return _save_and_view(game_id, state)


@app.post("/api/games/{game_id}/pass")
def submit_pass(game_id: str, body: PassRequest) -> dict:
    state = _get_state(game_id)
    try:
        cards = cards_from_ids(body.cards)
        state = state.submit_pass(store.human_player, cards)
        remaining_hand = list(state.hands[store.human_player])
        if state.phase.name == "PASSING":
            state = advance_bots(state, store.human_player)
        incoming = [
            card
            for card in state.hands[store.human_player]
            if card not in remaining_hand
        ]
    except (IllegalMoveError, ValueError) as exc:
        reason = exc.reason.name if isinstance(exc, IllegalMoveError) else "INVALID_PASS"
        raise HTTPException(
            status_code=400,
            detail={"reason": reason, "message": str(exc)},
        ) from exc
    return _save_and_view(
        game_id,
        state,
        action={
            "type": "pass",
            "player": store.human_player,
            "target": pass_target(store.human_player, state.pass_direction),
            "source": next(
                player
                for player in range(NUM_PLAYERS)
                if pass_target(player, state.pass_direction) == store.human_player
            ),
            "routes": [
                {
                    "source": player,
                    "target": pass_target(player, state.pass_direction),
                }
                for player in range(NUM_PLAYERS)
            ],
            "cards": cards,
            "incoming": incoming,
        },
    )


@app.post("/api/games/{game_id}/play")
def play_card(game_id: str, body: PlayRequest) -> dict:
    state = _get_state(game_id)
    try:
        card = card_from_id(body.card)
        trick_before = list(state.current_trick)
        state = state.play_card(store.human_player, card)
    except (IllegalMoveError, ValueError) as exc:
        reason = exc.reason.name if isinstance(exc, IllegalMoveError) else "ILLEGAL_CARD"
        raise HTTPException(
            status_code=400,
            detail={"reason": reason, "message": str(exc)},
        ) from exc
    trick_complete = len(trick_before) + 1 == NUM_PLAYERS
    return _save_and_view(
        game_id,
        state,
        action={
            "type": "play",
            "player": store.human_player,
            "card": card,
            "trick_winner": state.trick_leader if trick_complete else None,
            "trick_cards": trick_before + [(store.human_player, card)],
        },
    )


@app.post("/api/games/{game_id}/tick")
def tick_bot(game_id: str) -> dict:
    state = _get_state(game_id)
    state, action = advance_one_bot(state, store.human_player)
    return _save_and_view(game_id, state, action=action)


@app.post("/api/games/{game_id}/next-round")
def next_round(game_id: str) -> dict:
    state = _get_state(game_id)
    try:
        state = state.deal_new_round()
    except IllegalMoveError as exc:
        raise HTTPException(
            status_code=400,
            detail={"reason": exc.reason.name, "message": str(exc)},
        ) from exc
    return _save_and_view(game_id, state)


if WEB_DIR.exists():
    app.mount("/css", StaticFiles(directory=WEB_DIR / "css"), name="css")
    app.mount("/js", StaticFiles(directory=WEB_DIR / "js"), name="js")
    assets_dir = WEB_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")
