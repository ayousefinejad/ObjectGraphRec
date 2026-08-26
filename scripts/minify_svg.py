#!/usr/bin/env python
"""Shrink a matplotlib SVG so it fits comfortably in Notion's 200 KiB inline-content limit.

Notion accepts local images only as SVG passed inline as UTF-8 text, so every byte of a figure
has to travel through a tool call. Matplotlib emits ~6 decimal places per coordinate and repeats
the full DejaVu font-family fallback list on every text element, neither of which survives being
rendered at screen size.

Purely cosmetic: geometry is rounded to 0.1pt (well under a pixel at any sane zoom) and no
element is removed except the RDF metadata block.

    python scripts/minify_svg.py docs/notion/images/fig14_micro_graph_mode.svg
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# The full fallback list matplotlib writes out; a single generic family renders the same here.
FONT_RE = re.compile(r"font-family: '[^']*'(?:, '[^']*')*, sans-serif")
METADATA_RE = re.compile(r"<metadata>.*?</metadata>\s*", re.DOTALL)
NUM_RE = re.compile(r"\d+\.\d{2,}")


def _round(m: re.Match) -> str:
    return f"{float(m.group()):.1f}".removesuffix(".0")


def minify(svg: str) -> str:
    svg = METADATA_RE.sub("", svg)
    svg = FONT_RE.sub("font-family: sans-serif", svg)
    # Round coordinates to 0.1 -- but ONLY inside attribute values. Applying this to the whole
    # document also rewrites text nodes, which silently turns the axis label 0.05063 into 0.1
    # while leaving the element count identical, so a structural diff will not catch it.
    svg = re.sub(r'="([^"]*)"', lambda m: '="' + NUM_RE.sub(_round, m.group(1)) + '"', svg)
    return re.sub(r"\n\s*\n", "\n", svg)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--suffix", default=".min.svg", help="output suffix (default: .min.svg)")
    args = p.parse_args()

    for src in args.paths:
        out = src.with_suffix("").with_suffix(args.suffix) if src.suffix == ".svg" else src
        original = src.read_text()
        small = minify(original)
        out.write_text(small)
        print(f"{src.name}: {len(original.encode())/1024:.1f} KiB -> "
              f"{len(small.encode())/1024:.1f} KiB  ({100*len(small)/len(original):.0f}%)  {out.name}")


if __name__ == "__main__":
    main()
