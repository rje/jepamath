"""Side-by-side table of locate_surprise results: % of seeds where the twin's ending is further from the shared
prediction than the possible ending, inside the changed region (above 50% = the model expected the possible ending).

    python scripts/compare_locate.py NAME=path/locate_X.json [NAME=path ...] [--md]
"""
from __future__ import annotations

import argparse
import json

GROUPS = [("imp=add", "hidden addition"), ("delete,untouched", "hidden deletion, untouched object"),
          ("delete,handled", "hidden deletion, claw handled objects"), ("imp=delete", "hidden deletion, all"),
          ("ctl=move", "control: hidden move"), ("ctl=swap", "control: hidden swap")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=path to a locate_*.json")
    ap.add_argument("--md", action="store_true", help="markdown table")
    a = ap.parse_args()
    cols = []
    for spec in a.runs:
        name, path = spec.split("=", 1)
        cols.append((name, json.load(open(path))["summary"]))
    if a.md:
        print("| | " + " | ".join(n for n, _ in cols) + " |")
        print("|---|" + "---|" * len(cols))
    else:
        print(f"{'':<40}" + "".join(f"{n:>16}" for n, _ in cols))
    for key, label in GROUPS:
        cells = []
        for _, s in cols:
            if key not in s:
                raise KeyError(f"group {key!r} missing from a summary (have {list(s)})")
            v = s[key]
            lam = f" \u03bb {v['lam_mean']:.2f}" if "lam_mean" in v else ""
            lam += f"/{v['lam_pool']:.2f}" if "lam_pool" in v else ""
            cells.append(f"{v['twin>pos_in']:.0f}%{lam}")
        print(("| " + label + " | " + " | ".join(cells) + " |") if a.md else f"{label:<40}" + "".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
