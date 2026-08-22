import re
import xml.etree.ElementTree as ET
from pathlib import Path

text = Path("web/assets/svg-cards-ref.svg").read_text(encoding="utf-8")
ids = re.findall(r'id="([^"]+)"', text)
for needle in ("club", "spade", "heart", "diamond"):
    matches = [i for i in ids if needle in i.lower()]
    print(needle, matches[:15])

tree = ET.parse("web/assets/svg-cards-ref.svg")
root = tree.getroot()

def dump_group(gid: str) -> None:
    for g in root.iter():
        if g.get("id") == gid:
            print(f"\n=== {gid} ===")
            print(ET.tostring(g, encoding="unicode")[:2000])
            return

for gid in ("club", "spade", "heart", "diamond"):
    dump_group(gid)
