"""Re-score Track B evals with the scene-content confound removed (CPU, from saved eval_raw_*.pt).

Raw imp>ctl compares an impossible clip with its control, but they end with different numbers of objects
on screen (a hidden delete shows one fewer), and post-reveal surprise depends on how many objects are
visible. Here each clip's post-reveal surprise is first expressed relative to the mean surprise of
*possible* clips showing the same number of objects (a per-count mean, no fitted curve shape). The
impossible clip is then compared with its control on those residuals.

    python scripts/rescore_content.py RUN_DIR [RUN_DIR ...] --out OUT_DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import numpy as np
import torch


def post_scores(raw: dict, kind: str) -> np.ndarray:
    return np.array([float(v[j:].mean()) for v, j in zip(raw["slots"][kind], raw["first_post_slot"])])


def shown_counts(raw: dict) -> dict[str, np.ndarray]:
    final = np.array(raw["final"])
    viol = np.array(raw["imp_violation"])
    if not set(viol) <= {"add", "delete"}:
        raise ValueError(f"unexpected impossible violations {set(viol)}")
    return {"possible": final, "control": final, "impossible": final + np.where(viol == "add", 1, -1)}


def count_curve(score: np.ndarray, k: np.ndarray, min_n: int) -> dict[int, float]:
    return {int(c): float(score[k == c].mean()) for c in np.unique(k) if (k == c).sum() >= min_n}


def rescore(raw: dict, min_n: int = 10) -> dict:
    S = {kind: post_scores(raw, kind) for kind in ("possible", "impossible", "control")}
    K = shown_counts(raw)
    curve = count_curve(S["possible"], K["possible"], min_n)
    ok = np.array([K["impossible"][i] in curve and K["control"][i] in curve for i in range(len(S["possible"]))])
    res = {kind: np.array([S[kind][i] - curve.get(int(K[kind][i]), math.nan) for i in range(len(ok))])
           for kind in S}
    viol = np.array(raw["imp_violation"])

    def cmp(a, b, m):
        d = a[m] - b[m]
        return {"n": int(m.sum()), "acc": float((d > 0).mean() * 100), "diff": float(d.mean()),
                "sem": float(d.std(ddof=1) / math.sqrt(len(d)))}

    out = {"n_used": int(ok.sum()), "n_total": len(ok), "curve": curve}
    pairs = {"imp>ctl": ("impossible", "control"), "imp>pos": ("impossible", "possible"),
             "ctl>pos": ("control", "possible")}
    for pname, (a, b) in pairs.items():
        out[pname] = {"raw": {}, "resid": {}}
        for name, m in (("all", ok), ("add", ok & (viol == "add")), ("delete", ok & (viol == "delete"))):
            out[pname]["raw"][name] = cmp(S[a], S[b], m)
            out[pname]["resid"][name] = cmp(res[a], res[b], m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", type=int, default=10, help="fewest possible clips at a count to trust its mean")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for run in a.runs:
        files = sorted(glob.glob(os.path.join(run, "eval_raw_*.pt")))
        if not files:
            raise FileNotFoundError(f"no eval_raw_*.pt in {run}")
        for f in files:
            step = int(re.search(r"eval_raw_(\d+)\.pt", f)[1])
            raw = torch.load(f, map_location="cpu", weights_only=False)
            if "slots" not in raw:
                raise KeyError(f"{f}: no 'slots' (keys {list(raw)})")
            r = rescore(raw, a.min_n)
            rows.append({"run": os.path.basename(run.rstrip("/")), "step": step, **r})
            print(f"{rows[-1]['run']:<18} {step:>6}  " + "  ".join(
                f"{p} raw/resid add {r[p]['raw']['add']['acc']:4.1f}/{r[p]['resid']['add']['acc']:4.1f} "
                f"del {r[p]['raw']['delete']['acc']:4.1f}/{r[p]['resid']['delete']['acc']:4.1f}"
                for p in ("imp>ctl", "imp>pos", "ctl>pos")))
    with open(os.path.join(a.out, "rescore_content.json"), "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
