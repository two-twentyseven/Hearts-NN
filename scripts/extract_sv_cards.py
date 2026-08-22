"""Extract individual card SVGs from SVG-cards 4.0 sprite sheet."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "svg-cards-source.svg"
OUT_DIR = Path(__file__).resolve().parents[1] / "web" / "assets" / "cards"

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

VIEW_W = 169.075
VIEW_H = 244.640

RANK_MAP = {
    "1": "A", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
    "8": "8", "9": "9", "10": "T", "jack": "J", "queen": "Q", "king": "K",
}
SUIT_MAP = {"club": "C", "diamond": "D", "heart": "H", "spade": "S"}

CARD_IDS: list[str] = []
for suit in ("club", "diamond", "heart", "spade"):
    for rank in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king"):
        CARD_IDS.append(f"{suit}_{rank}")


def local_name(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def find_by_id(root: ET.Element, element_id: str) -> ET.Element | None:
    for elem in root.iter():
        if elem.get("id") == element_id:
            return elem
    return None


def collect_hrefs(elem: ET.Element) -> set[str]:
    refs: set[str] = set()
    href = elem.get(f"{{{XLINK_NS}}}href") or elem.get("href")
    if href and href.startswith("#"):
        refs.add(href[1:])
    for child in elem:
        refs |= collect_hrefs(child)
    return refs


def prune_defs(defs_elem: ET.Element, needed_ids: set[str]) -> None:
    all_children = list(defs_elem)
    by_id = {child.get("id"): child for child in all_children if child.get("id")}

    expanded = set(needed_ids)
    changed = True
    while changed:
        changed = False
        for ref_id in list(expanded):
            elem = by_id.get(ref_id)
            if elem is None:
                continue
            for sub_ref in collect_hrefs(elem):
                if sub_ref not in expanded:
                    expanded.add(sub_ref)
                    changed = True

    for child in all_children:
        child_id = child.get("id")
        if child_id and child_id not in expanded:
            defs_elem.remove(child)


def strip_use_stretch_attrs(root: ET.Element) -> None:
    """Remove width/height on <use> — they stretch symbols to the full card."""
    for elem in root.iter():
        if local_name(elem.tag) != "use":
            continue
        for attr in ("width", "height"):
            if elem.get(attr) in ("100%", str(VIEW_W), str(VIEW_H)):
                del elem.attrib[attr]


def fix_rank_text(defs_elem: ET.Element) -> None:
    """Use uniform scale so corner ranks are not vertically stretched."""
    for elem in defs_elem.iter():
        if local_name(elem.tag) != "text":
            continue
        transform = elem.get("transform", "")
        if "scale(0.56453" in transform:
            elem.set("transform", "scale(0.72)")


def fix_card_base(defs_elem: ET.Element) -> None:
    """Slightly thinner border; fill-only avoids double-edge artifacts when clipped."""
    base = find_by_id(defs_elem, "base")
    if base is None:
        return
    for path in base.iter():
        if local_name(path.tag) != "path":
            continue
        style = path.get("style", "")
        style = style.replace("stroke-width:2.5", "stroke-width:1.8")
        path.set("style", style)


def card_to_svg(card_elem: ET.Element, defs_elem: ET.Element | None) -> str:
    svg = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "xmlns": SVG_NS,
            "xmlns:xlink": XLINK_NS,
            "width": str(VIEW_W),
            "height": str(VIEW_H),
            "viewBox": f"0 0 {VIEW_W} {VIEW_H}",
        },
    )

    card_copy = deepcopy(card_elem)
    strip_use_stretch_attrs(card_copy)

    if defs_elem is not None:
        needed = collect_hrefs(card_copy)
        defs_copy = deepcopy(defs_elem)
        prune_defs(defs_copy, needed)
        fix_rank_text(defs_copy)
        fix_card_base(defs_copy)
        svg.append(defs_copy)

    svg.append(card_copy)
    return ET.tostring(svg, encoding="unicode")


def to_game_id(card_id: str) -> str | None:
    if card_id == "alternate-back":
        return "back"
    parts = card_id.split("_", 1)
    if len(parts) != 2:
        return None
    suit, rank = parts
    if suit not in SUIT_MAP or rank not in RANK_MAP:
        return None
    return f"{RANK_MAP[rank]}{SUIT_MAP[suit]}"


def main() -> None:
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    defs_elem = find_by_id(root, "defs") or root.find(f"{{{SVG_NS}}}defs")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    exported = 0
    for card_id in CARD_IDS + ["alternate-back"]:
        elem = find_by_id(root, card_id)
        if elem is None:
            print(f"Missing: {card_id}")
            continue
        game_id = to_game_id(card_id)
        if game_id is None:
            continue
        svg_text = card_to_svg(elem, defs_elem)
        (OUT_DIR / f"{game_id}.svg").write_text(svg_text, encoding="utf-8")
        exported += 1
        print(f"  {card_id} -> {game_id}.svg")

    print(f"Exported {exported} cards to {OUT_DIR}")


if __name__ == "__main__":
    main()
