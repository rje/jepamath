"""Verify the paired outputs and report the predeclared seed-0 gate."""

import argparse
import json
from pathlib import Path
from jepatrain.placement_eval import compare_runs


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
    result = dict(
        step=100000,
        scenes=128,
        visible_qualified=visible,
        treatment_add_pair=score,
        paired_gains=gain,
        real_add_gain=real_add_delta,
        retention=retention,
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
