"""Verify the paired outputs and report the predeclared seed-0 gate."""

import argparse
import json
from pathlib import Path

import numpy as np

from jepatrain.placement_eval import compare_runs, intervals


def visible_count(control, treatment):
    counts = [json.loads(p.read_text()) for p in (control, treatment)]
    meta = counts[0]["test_meta"]
    if meta != counts[1]["test_meta"] or len(meta["box"]) != 600 or any(c["step"] != 100000 for c in counts):
        raise ValueError("unpaired fixed count battery")
    result = {}
    for target in ("box", "total"):
        labels = np.asarray(meta[target])
        hits = np.array([np.asarray(c[f"xfmr/{target}/pred_int"]) == labels for c in counts], dtype=float)
        result[target] = dict(control=intervals(hits[0]), treatment=intervals(hits[1]), gain=intervals(hits[1]-hits[0]))
    return result


def main(control, treatment, out):
    a, b = [json.loads(p.read_text()) for p in (control, treatment)]
    if a["step"] != 100000 or a["split"] != "seed0" or len(a["rows"]) != 128:
        raise ValueError("incomplete endpoint")
    gain = compare_runs(control, treatment)
    arm = b["summary"]["main"]
    score = arm["carry"]["pairs"][0]
    real_add_delta = arm["carry"]["cells"][1][0] - a["summary"]["main"]["carry"]["cells"][1][0]
    visible = all(s[1] > 0.75 for d in (a, b) for s in d["summary"]["main"]["visible"]["cells"])
    retention = {
        group: min(x[0] for x in gain["main_" + stage])
        for group, stage in (("landed", "landed"), ("visible", "visible"))
    }
    retention["removal"] = gain["main_carry"][1][0]
    count = visible_count(*(p.parent / "visible-count-100000.json" for p in (control, treatment)))
    retention["box_count"] = count["box"]["gain"][0]
    result = dict(
        step=100000,
        scenes=128,
        visible_qualified=visible,
        treatment_add_pair=score,
        paired_gains=gain,
        real_add_gain=real_add_delta,
        fixed_visible_count=count,
        retention=retention,
        retention_pass=all(retention[k] >= -0.05 for k in ("landed", "removal", "box_count")),
        primary_pass=visible and score[1] > 0.60 and gain["main_carry"][0][1] > 0 and real_add_delta > 0,
        fork_treatment=b["summary"]["fork"]["carry"],
        fork_gain=gain["fork_carry"],
        control=a["summary"],
        treatment=b["summary"],
    )
    out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "treatment", "out"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    main(a.control, a.treatment, a.out)
