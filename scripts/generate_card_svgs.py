"""Generate clean SVG playing cards for the Hearts web UI.

Poker-size 5:7 face with a stacked corner index (rank over pip) and a
centered suit. Indices are large enough to read on the trick, inset so they
clear the rounded border, and centered as a column so "10" does not collide
with the pip or the edge.
"""

from __future__ import annotations

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "web" / "assets" / "cards"

W, H = 250, 350
RADIUS = 18
BORDER = 3
# Index column center and top; leave room inside the rounded-rect curve.
INDEX_X = 38
INDEX_Y = 26
CENTER_PIP = 92

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
FONT = "Arial,Helvetica Neue,Helvetica,sans-serif"


def corner(rank: str, suit: str, color: str, x: float, y: float, rotate: float) -> str:
    label = RANK_LABELS[rank]
    symbol, _ = SUITS[suit]
    wide = label == "10"
    rank_size = 36 if wide else 44
    suit_size = 30 if wide else 34
    suit_y = rank_size + 6
    transform = f'transform="translate({x},{y}) rotate({rotate})"'
    return (
        f'<g {transform}>'
        f'<text x="0" y="0" fill="{color}" font-family="{FONT}" font-size="{rank_size}" '
        f'font-weight="700" text-anchor="middle" dominant-baseline="hanging">{label}</text>'
        f'<text x="0" y="{suit_y}" fill="{color}" font-family="{FONT}" font-size="{suit_size}" '
        f'text-anchor="middle" dominant-baseline="hanging">{symbol}</text>'
        f"</g>"
    )


def card_svg(rank: str, suit: str) -> str:
    symbol, color = SUITS[suit]
    inset = BORDER / 2
    inner_w = W - BORDER
    inner_h = H - BORDER
    corners = (
        corner(rank, suit, color, INDEX_X, INDEX_Y, 0)
        + corner(rank, suit, color, W - INDEX_X, H - INDEX_Y, 180)
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
  <rect x="{inset}" y="{inset}" width="{inner_w}" height="{inner_h}" rx="{RADIUS}"
    fill="#ffffff" stroke="#b7b3ab" stroke-width="{BORDER}"/>
  {corners}
  <text x="{W / 2}" y="{H / 2}" fill="{color}" text-anchor="middle" dominant-baseline="middle"
    font-family="{FONT}" font-size="{CENTER_PIP}">{symbol}</text>
</svg>"""


def back_svg() -> str:
    inset = BORDER / 2
    inner = 16
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
  <rect x="{inset}" y="{inset}" width="{W - BORDER}" height="{H - BORDER}" rx="{RADIUS}"
    fill="#1e3a5f" stroke="#2d5080" stroke-width="{BORDER}"/>
  <rect x="{inner}" y="{inner}" width="{W - inner * 2}" height="{H - inner * 2}" rx="{RADIUS - 6}"
    fill="none" stroke="rgba(255,255,255,0.28)" stroke-width="2"/>
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
