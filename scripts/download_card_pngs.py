"""Download official SVG-cards PNG assets and trim black borders."""

from __future__ import annotations

import urllib.request
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Pillow is required: pip install Pillow") from exc

OUT_DIR = Path(__file__).resolve().parents[1] / "web" / "assets" / "cards"
BASE_URL = "https://raw.githubusercontent.com/htdebeer/SVG-cards/master/png/2x"
TRIM_PX = 10

RANK_MAP = {
    "1": "A", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
    "8": "8", "9": "9", "10": "T", "jack": "J", "queen": "Q", "king": "K",
}
SUIT_MAP = {"club": "C", "diamond": "D", "heart": "H", "spade": "S"}

SOURCE_NAMES: list[str] = []
for suit in ("club", "diamond", "heart", "spade"):
    for rank in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king"):
        SOURCE_NAMES.append(f"{suit}_{rank}")
SOURCE_NAMES.append("back")


def to_game_id(source: str) -> str:
    if source == "back":
        return "back"
    suit, rank = source.split("_", 1)
    return f"{RANK_MAP[rank]}{SUIT_MAP[suit]}"


def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Hearts-NN-asset-fetch"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def trim_card_border(data: bytes) -> Image.Image:
    from io import BytesIO

    img = Image.open(BytesIO(data)).convert("RGBA")
    w, h = img.size
    trimmed = img.crop((TRIM_PX, TRIM_PX, w - TRIM_PX, h - TRIM_PX))
    return trimmed


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.svg"):
        old.unlink()

    for source in SOURCE_NAMES:
        game_id = to_game_id(source)
        url = f"{BASE_URL}/{source}.png"
        dest = OUT_DIR / f"{game_id}.png"
        print(f"  {source}.png -> {dest.name}")
        raw = download(url)
        trimmed = trim_card_border(raw)
        trimmed.save(dest, format="PNG", optimize=True)

    sample = Image.open(OUT_DIR / "7C.png")
    print(f"Downloaded {len(SOURCE_NAMES)} cards to {OUT_DIR} ({sample.size[0]}x{sample.size[1]} px)")


if __name__ == "__main__":
    main()
