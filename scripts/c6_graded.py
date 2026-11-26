"""C6 graded analysis: does the predicted ending's 3-vs-1 preference move with the hidden count, and is it area?

Exploratory on c6_v2 (found after its pre-registered binary endpoints were at chance); the confirmation on fresh
items is pre-registered separately. Per block (one source's 7 histories vs one donor's 1/2/3-object endings):
  pref[h] = (S[h, 1-obj] - S[h, 3-obj]) / sd_h   (> 0: the prediction is closer to the 3-object ending)
  components: least squares pref ~ 1 + N + add + remove  ->  (bN, b_add, b_rem)
  slope:      least squares pref ~ 1 + final               ->  per-block slope
  area:       across blocks, slope ~ 1 + log(area3 / area1 on the block's common patches); the intercept is the
              slope at equal area. Also the mean slope over blocks where the 3-object ending has LESS area.
Block bootstrap (2,000; seed fixed) for every interval.

    python scripts/c6_graded.py --manifest M.json --mats DIR/c6_v2 --names objlabel-100k plain-100k ... --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from c6_items import CELLS  # noqa: E402

N_ = np.array([N for _, N, _, _ in CELLS], float)
A_ = np.array([a for _, _, a, _ in CELLS], float)
R_ = np.array([r for _, _, _, r in CELLS], float)
FIN = N_ + A_ - R_
XC = np.c_[np.ones(7), N_, A_, R_]
XF = np.c_[np.ones(7), FIN]


def prefs(M: np.ndarray) -> np.ndarray:
    p = M[:, :, 0] - M[:, :, 2]
    return p / (p.std(1, keepdims=True) + 1e-12)


def area_log_ratio(man: dict) -> np.ndarray:
    out = []
    for p in man["pairs"]:
        cov = np.load(os.path.join(p["donor_dir"], f"donor_{p['donor']:09d}.npz"))["coverage"].reshape(3, -1)
        cov = cov[:, p["common"]].sum(1)
        out.append(np.log(cov[2] / cov[0]))
    return np.array(out)


def ci(boot: np.ndarray) -> list:
    return np.percentile(boot, [2.5, 97.5], 0).T.tolist()


def analyse(M: np.ndarray, adiff: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    P = prefs(M)
    comp = np.array([np.linalg.lstsq(XC, p, rcond=None)[0][1:] for p in P])
    slope = np.array([np.linalg.lstsq(XF, p, rcond=None)[0][1] for p in P])
    A = np.c_[np.ones(len(slope)), adiff]
    fit = np.linalg.lstsq(A, slope, rcond=None)[0]
    bc, bf, bl = [], [], []
    less = adiff < 0
    for _ in range(n_boot):
        i = rng.integers(0, len(P), len(P))
        bc.append(comp[i].mean(0))
        bf.append(np.linalg.lstsq(A[i], slope[i], rcond=None)[0])
        j = i[less[i]]
        bl.append(slope[j].mean())
    bc, bf, bl = np.array(bc), np.array(bf), np.array(bl)
    return {"n_blocks": len(P),
            "components": {k: {"est": float(comp[:, n].mean()), "ci": ci(bc)[n]} for n, k in enumerate(("N", "add", "remove"))},
            "slope_at_equal_area": {"est": float(fit[0]), "ci": ci(bf)[0]},
            "slope_per_log_area": {"est": float(fit[1]), "ci": ci(bf)[1]},
            "slope_where_3_has_less_area": {"est": float(slope[less].mean()), "ci": np.percentile(bl, [2.5, 97.5]).tolist(),
                                            "n": int(less.sum())},
            "choice_share_1_2_3": (np.bincount(M.argmin(2).ravel(), minlength=3) / M[:, :, 0].size).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mats", required=True, help="prefix: <prefix>_<name>_mats.npz")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = json.load(open(a.manifest))
    adiff = area_log_ratio(man)
    res = {"manifest": a.manifest, "area_log_ratio": {"mean": float(adiff.mean()), "sd": float(adiff.std())}, "models": {}}
    for nm in a.names:
        z = np.load(f"{a.mats}_{nm}_mats.npz")
        if z["source"].tolist() != [p["source"] for p in man["pairs"]]:
            raise SystemExit(f"{nm}: block order differs from the manifest")
        r = res["models"][nm] = analyse(z["mats"], adiff)
        c = r["components"]
        print(f"{nm:16s} N {c['N']['est']:+.3f}  add {c['add']['est']:+.3f}  rem {c['remove']['est']:+.3f}  "
              f"slope@equal-area {r['slope_at_equal_area']['est']:+.3f} {np.round(r['slope_at_equal_area']['ci'], 3).tolist()}  "
              f"slope|3 has less area {r['slope_where_3_has_less_area']['est']:+.3f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
