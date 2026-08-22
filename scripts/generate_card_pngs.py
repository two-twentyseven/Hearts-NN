"""Generate minimal playing-card PNGs for the Hearts web UI.

Clean white cards: bold rank + suit in corners, large suit in center.
"""

from __future__ import annotations

from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit("Pillow is required: pip install Pillow") from exc

OUT_DIR = Path(__file__).resolve().parents[1] / "web" / "assets" / "cards"

W, H = 318, 469
RADIUS = 18
PAD = 22

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
RANK_LABELS = {
    "A": "A", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6",
    "7": "7", "8": "8", "9": "9", "T": "10", "J": "J", "Q": "Q", "K": "K",
}

SUITS = {
    "H": ("♥", (196, 30, 58)),
    "D": ("♦", (196, 30, 58)),
    "S": ("♠", (26, 26, 26)),
    "C": ("♣", (26, 26, 26)),
}

FONT_CANDIDATES = {
    "regular": [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ],
    "bold": [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ],
    "symbol": [
        "C:/Windows/Fonts/seguisym.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
}


def load_font(kind: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES[kind]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_corner_block(
    rank: str,
    suit: str,
    color: tuple[int, int, int],
) -> Image.Image:
    label = RANK_LABELS[rank]
    symbol, _ = SUITS[suit]

    rank_size = 38 if label != "10" else 30
    suit_size = 30 if label != "10" else 26

    rank_font = load_font("bold", rank_size)
    suit_font = load_font("symbol", suit_size)

    block_w = 72
    block_h = rank_size + suit_size + 8
    block = Image.new("RGBA", (block_w, block_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(block)
    draw.text((0, 0), label, fill=color, font=rank_font)
    draw.text((0, rank_size + 2), symbol, fill=color, font=suit_font)
    return block.crop(block.getbbox())


def draw_card(rank: str, suit: str) -> Image.Image:
    symbol, color = SUITS[suit]

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        (1, 1, W - 2, H - 2),
        radius=RADIUS,
        fill=(255, 255, 255, 255),
        outline=(200, 196, 188, 255),
        width=2,
    )

    corner = draw_corner_block(rank, suit, color)
    img.paste(corner, (PAD, PAD), corner)

    inverted = corner.rotate(180, expand=True)
    img.paste(inverted, (W - PAD - inverted.width, H - PAD - inverted.height), inverted)

    center_font = load_font("symbol", 112)
    tw, th = text_size(draw, symbol, center_font)
    draw.text(
        ((W - tw) / 2, (H - th) / 2 - 6),
        symbol,
        fill=color,
        font=center_font,
    )

    return img


def draw_back() -> Image.Image:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        (1, 1, W - 2, H - 2),
        radius=RADIUS,
        fill=(30, 58, 95, 255),
        outline=(45, 78, 120, 255),
        width=2,
    )
    draw.rounded_rectangle(
        (16, 16, W - 17, H - 17),
        radius=RADIUS - 4,
        outline=(255, 255, 255, 48),
        width=2,
    )
    for y in range(24, H - 24, 14):
        draw.line((24, y, W - 24, y), fill=(255, 255, 255, 22), width=1)
    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for rank in RANKS:
        for suit in SUITS:
            card_id = f"{rank}{suit}"
            draw_card(rank, suit).save(OUT_DIR / f"{card_id}.png", format="PNG", optimize=True)

    draw_back().save(OUT_DIR / "back.png", format="PNG", optimize=True)
    print(f"Generated {len(RANKS) * len(SUITS) + 1} PNGs in {OUT_DIR} ({W}x{H})")


if __name__ == "__main__":
    main()
