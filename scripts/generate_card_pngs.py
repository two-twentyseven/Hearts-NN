"""Generate minimal playing-card PNGs for the Hearts web UI.

Matches the SVG layout: inset rounded border, stacked corner index, centered pip.
"""

from __future__ import annotations

from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit("Pillow is required: pip install Pillow") from exc

OUT_DIR = Path(__file__).resolve().parents[1] / "web" / "assets" / "cards"

W, H = 500, 700
RADIUS = 36
BORDER = 6
INDEX_X = 76
INDEX_Y = 52

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
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "bold": [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "symbol": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "C:/Windows/Fonts/seguisym.ttf",
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
    wide = label == "10"
    rank_size = 64 if wide else 80
    suit_size = 52 if wide else 60

    rank_font = load_font("bold", rank_size)
    suit_font = load_font("symbol", suit_size)

    scratch = Image.new("RGBA", (160, rank_size + suit_size + 24), (0, 0, 0, 0))
    draw = ImageDraw.Draw(scratch)
    rw, _ = text_size(draw, label, rank_font)
    sw, _ = text_size(draw, symbol, suit_font)
    col_w = max(rw, sw)
    draw.text(((col_w - rw) / 2, 0), label, fill=color, font=rank_font)
    draw.text(((col_w - sw) / 2, rank_size + 8), symbol, fill=color, font=suit_font)
    bbox = scratch.getbbox()
    return scratch.crop(bbox) if bbox else scratch


def draw_card(rank: str, suit: str) -> Image.Image:
    symbol, color = SUITS[suit]

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    inset = BORDER / 2
    draw.rounded_rectangle(
        (inset, inset, W - 1 - inset, H - 1 - inset),
        radius=RADIUS,
        fill=(255, 255, 255, 255),
        outline=(183, 179, 171, 255),
        width=BORDER,
    )

    corner = draw_corner_block(rank, suit, color)
    img.paste(corner, (INDEX_X - corner.width // 2, INDEX_Y), corner)

    inverted = corner.rotate(180, expand=True)
    img.paste(
        inverted,
        (W - INDEX_X - inverted.width // 2, H - INDEX_Y - inverted.height),
        inverted,
    )

    center_font = load_font("symbol", 168)
    tw, th = text_size(draw, symbol, center_font)
    draw.text(
        ((W - tw) / 2, (H - th) / 2 - 8),
        symbol,
        fill=color,
        font=center_font,
    )
    return img


def draw_back() -> Image.Image:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    inset = BORDER / 2
    draw.rounded_rectangle(
        (inset, inset, W - 1 - inset, H - 1 - inset),
        radius=RADIUS,
        fill=(30, 58, 95, 255),
        outline=(45, 78, 120, 255),
        width=BORDER,
    )
    inner = 32
    draw.rounded_rectangle(
        (inner, inner, W - inner - 1, H - inner - 1),
        radius=RADIUS - 10,
        outline=(255, 255, 255, 72),
        width=3,
    )
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
