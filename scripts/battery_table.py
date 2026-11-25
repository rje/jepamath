"""Results table from the battery JSONs (scripts/eval_battery.py): one row per checkpoint, value [95% CI].

    python scripts/battery_table.py [DIR] > table.md
"""
import glob
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/jepamath/trackb/battery"
COLS = [("Grouping AUC", "object", "grouping", 1, 2), ("Moved id %", "object", "identity_moved", 100, 0),
        ("Kind %", "object", "kind", 100, 0), ("Size R²", "object", "size_r2", 1, 2),
        ("Depth R²", "object", "depth3d_r2", 1, 2), ("Height R²", "object", "height3d_r2", 1, 2),
        ("Box %", "count", "box", 1, 1), ("Total %", "count", "total", 1, 1),
        ("Area − size", "count", "area_minus_size", 1, 1), ("Still coef", "count", "coef_still", 1, 2),
        ("Dark drop", "count_dark", "box_drop", 1, 1), ("Small drop", "count_small", "box_drop", 1, 1)]


def cell(sec, k, s, p):
    v = sec[k]; lo, hi = sec[k + "_ci"]
    return f"{v * s:.{p}f} [{lo * s:.{p}f}, {hi * s:.{p}f}]"


rows = []
for f in sorted(glob.glob(os.path.join(d, "*.json"))):
    b = json.load(open(f))
    secs = {"object": b["object"]["8"], "count": b["count"], "count_dark": b["count_dark"], "count_small": b["count_small"]}
    v = b["voe"]
    h = (f"{v['delete,handled']['pct']:.0f} / {v['delete,untouched']['pct']:.0f}" if "delete,handled" in v else "–")
    rows.append([b["name"], str(b["step"] // 1000) + "k"] + [cell(secs[s], k, sc, p) for _, s, k, sc, p in COLS] + [h])
hdr = ["Checkpoint", "Step"] + [c[0] for c in COLS] + ["VoE handled / untouched %"]
print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
for r in rows:
    print("| " + " | ".join(r) + " |")
print(f"\nLayer 8, 95% bootstrap CIs over clips; {len(rows)} checkpoints. Layer-0 baselines are in each JSON's object['0'].")
