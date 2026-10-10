"""Print an SVG's element tree with long attribute values truncated.

Lets the structure of the generated logo files be inspected without dumping
the multi-kilobyte path data.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from xml.etree import ElementTree as ET

NS = "{http://www.w3.org/2000/svg}"


def short(value: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[:limit] + f"...[{len(value)} chars]"


def walk(node: ET.Element, depth: int, limit: int) -> None:
    tag = node.tag.replace(NS, "")
    attrs = " ".join(f'{k}="{short(v, limit)}"' for k, v in node.attrib.items())
    text = short(node.text or "", limit)
    body = f" text={text!r}" if text else ""
    print(f"{'  ' * depth}<{tag} {attrs}>{body}")
    for child in node:
        walk(child, depth + 1, limit)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("svg")
    ap.add_argument("--limit", type=int, default=90)
    ap.add_argument("--skip", default="", help="comma separated tags to collapse")
    args = ap.parse_args()

    skip = {t for t in args.skip.split(",") if t}
    root = ET.parse(Path(args.svg)).getroot()

    def walk_filtered(node: ET.Element, depth: int) -> None:
        tag = node.tag.replace(NS, "")
        if tag in skip:
            return
        attrs = " ".join(f'{k}="{short(v, args.limit)}"' for k, v in node.attrib.items())
        text = short(node.text or "", args.limit)
        body = f" text={text!r}" if text else ""
        kids = [c for c in node if c.tag.replace(NS, "") in skip]
        note = f"   [+{len(kids)} skipped {','.join(sorted({c.tag.replace(NS, '') for c in kids}))}]" if kids else ""
        print(f"{'  ' * depth}<{tag} {attrs}>{body}{note}")
        for child in node:
            walk_filtered(child, depth + 1)

    walk_filtered(root, 0)


if __name__ == "__main__":
    main()
