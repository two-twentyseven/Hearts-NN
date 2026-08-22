"""Generate clean minimal SVG playing cards for the Hearts web UI."""

from __future__ import annotations

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "web" / "assets" / "cards"

W, H = 100, 140
RADIUS = 6
PAD = 7

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
RANK_LABELS = {
    "A": "A", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
    "7": "7", "8": "8", "9": "9", "T": "10", "J": "J", "Q": "Q", "K": "K",
}
SUITS = {
    "H": ("♥", "#c41e3a"),
    "D": ("♦", "#c41e3a"),
    "S": ("♠", "#1a1a1a"),
    "C": ("♣", "#1a1a1a"),
}


def corner(rank: str, suit: str, color: str, x: float, y: float, rotate: float) -> str:
    label = RANK_LABELS[rank]
    symbol, _ = SUITS[suit]
    font_size = 13 if label != "10" else 11
    suit_y = font_size + 1
    transform = f'transform="translate({x},{y}) rotate({rotate})"'
    return (
        f'<g {transform}>'
        f'<text x="0" y="{font_size}" fill="{color}" font-family="Segoe UI,Arial,sans-serif" '
        f'font-size="{font_size}" font-weight="700">{label}</text>'
        f'<text x="0" y="{suit_y + 11}" fill="{color}" font-family="Segoe UI Symbol,Segoe UI,Arial,sans-serif" '
        f'font-size="11">{symbol}</text>'
        f"</g>"
    )


def card_svg(rank: str, suit: str) -> str:
    symbol, color = SUITS[suit]
    corners = (
        corner(rank, suit, color, PAD, PAD, 0)
        + corner(rank, suit, color, W - PAD, H - PAD, 180)
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
  <rect width="{W}" height="{H}" rx="{RADIUS}" fill="#ffffff" stroke="#c8c4bc" stroke-width="0.75"/>
  {corners}
  <text x="{W / 2}" y="{H / 2 + 6}" fill="{color}" text-anchor="middle" dominant-baseline="middle"
    font-family="Segoe UI Symbol,Segoe UI,Arial,sans-serif" font-size="36">{symbol}</text>
</svg>"""


def back_svg() -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
  <rect width="{W}" height="{H}" rx="{RADIUS}" fill="#1e3a5f" stroke="#2d5080" stroke-width="0.75"/>
  <rect x="6" y="6" width="{W - 12}" height="{H - 12}" rx="5" fill="none" stroke="rgba(255,255,255,0.25)" stroke-width="0.75"/>
</svg>"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for rank in RANKS:
        for suit in SUITS:
            card_id = f"{rank}{suit}"
            (OUT_DIR / f"{card_id}.svg").write_text(card_svg(rank, suit), encoding="utf-8")
    (OUT_DIR / "back.svg").write_text(back_svg(), encoding="utf-8")
    print(f"Generated {len(RANKS) * len(SUITS) + 1} SVGs in {OUT_DIR}")


if __name__ == "__main__":
    main()
