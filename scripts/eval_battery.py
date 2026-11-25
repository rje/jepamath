"""The fixed measurement battery (research program, S0): one JSON per checkpoint, every metric with a 95% CI.

Runs, at the claim layer (8) with the layer-0 baseline where it applies:
  object sense : scripts/object_sense.py (grouping, hard grouping, identity, moved identity, figure, kind, size, 3D),
                 bootstrap over clips
  count        : scripts/count_by_n.py on v4b_probe (transformer readout, fit 1400 / test 600): box and total
                 accuracy, area-matched vs size-matched box accuracy and their difference, the moved-vs-still
                 coefficients; the same readout fit on v4b_probe and tested on v4b_probe_dark / v4b_probe_small;
                 extrapolation (linear-mean readout fit on box <= 2 / total <= 4). Bootstrap over test clips
  VoE          : read from the run's own locate/ and visdrop/ JSONs for this step (computed in the cloud job). CIs:
                 Wilson for the twin > possible percentages, mean +- 1.96 sem for lambda. Absent JSONs are recorded
                 as null with the reason; they are never filled in

    CUDA_VISIBLE_DEVICES= python scripts/eval_battery.py RUN --weights W --name NAME [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import object_sense  # noqa: E402
from jepatrain.reeval import load_run  # noqa: E402

PROBE = "/mnt/data/jepamath/v4b_probe"
LAYER = 8
VOE_KEYS = ("delete,handled", "delete,untouched", "imp=add", "ctl=move")


def ci(v):
    v = np.asarray(v, dtype=float)
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def wilson(pct, n):
    p, z = pct / 100, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [100 * (c - h), 100 * (c + h)]


def count_run(run, weights, out, *extra):
    cmd = [sys.executable, os.path.join(HERE, "count_by_n.py"), run, "--weights", weights, "--layer", str(LAYER),
           "--out", out, *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stderr[-3000:]}")
    return json.load(open(out))


def count_metrics(c, boot, rng, readout="xfmr"):
    """Accuracies and controls from one count_by_n JSON, with bootstrap CIs over its test clips."""
    m = c["test_meta"]
    box, tot, mode, moved = (np.array(m[k]) for k in ("box", "total", "mode", "moved"))
    pb, pt = np.array(c[f"{readout}/box/pred_int"]), np.array(c[f"{readout}/total/pred_int"])
    pbr = np.array(c[f"{readout}/box/pred"])
    am, sm = mode == "area_matched", mode == "size_matched"
    if not am.any() or not sm.any():
        raise ValueError(f"probe set lacks area_matched or size_matched clips: {sorted(set(mode))}")

    def f(idx):
        hb, ht = pb[idx] == box[idx], pt[idx] == tot[idx]
        a_am, a_sm = hb[am[idx]].mean() * 100, hb[sm[idx]].mean() * 100
        A = np.c_[moved[idx], box[idx] - moved[idx], np.ones(len(idx))]
        coef = np.linalg.lstsq(A, pbr[idx], rcond=None)[0]
        return {"box": hb.mean() * 100, "total": ht.mean() * 100, "box_area_matched": a_am, "box_size_matched": a_sm,
                "area_minus_size": a_am - a_sm, "coef_moved": coef[0], "coef_still": coef[1]}

    n = len(box)
    out = {k: float(v) for k, v in f(np.arange(n)).items()}
    draws = [f(rng.integers(0, n, n)) for _ in range(boot)]
    for k in list(out):
        out[f"{k}_ci"] = ci([d[k] for d in draws])
    out["n_test"] = n
    return out, draws


def voe(run, step):
    out = {}
    loc = os.path.join(run, "locate", f"locate_run_{step:06d}.json")
    if os.path.exists(loc):
        s = json.load(open(loc))["summary"]
        for k in VOE_KEYS:
            out[k] = {"pct": s[k]["twin>pos_in"], "n": s[k]["n"], "ci": wilson(s[k]["twin>pos_in"], s[k]["n"])}
    else:
        out["locate_missing"] = loc
    vis = os.path.join(run, "visdrop", f"visdrop_run_{step:06d}.json")
    if os.path.exists(vis):
        s = json.load(open(vis))["summary"]
        for k in ("held/all", "landed/all", "frozen/all"):
            mu, se = s[k]["lam_mean"], s[k]["lam_sem"]
            out[f"lambda_{k.split('/')[0]}"] = {"mean": mu, "n": s[k]["n"], "ci": [mu - 1.96 * se, mu + 1.96 * se]}
    else:
        out["visdrop_missing"] = vis
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--weights", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="/mnt/data/jepamath/trackb/battery")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("run with CUDA_VISIBLE_DEVICES=")
    torch.set_num_threads(40)
    os.makedirs(os.path.join(a.out_dir, "raw"), exist_ok=True)
    raw = lambda tag: os.path.join(a.out_dir, "raw", f"{a.name}_{tag}.json")  # noqa: E731
    enc, _t, _p, step = load_run(a.run, a.weights, device="cpu")
    enc.eval()
    if len(enc.blocks) != LAYER:
        raise ValueError(f"battery claims are at layer {LAYER}; this encoder has {len(enc.blocks)} blocks")
    res = json.load(open(os.path.join(a.run, "args.json")))["res"]
    if res != 128:
        raise ValueError(f"battery probe sets are fixed at 128 px; run is {res}")
    out = {"name": a.name, "run": a.run, "weights": a.weights, "step": step, "layer": LAYER, "boot": a.boot}

    print(f"[{a.name}] object sense", flush=True)
    r = object_sense.collect(enc, res, PROBE, 200, [0, LAYER])
    out["object"] = {str(L): object_sense.score(r, L, a.boot) for L in (0, LAYER)}
    del enc, r

    rng = np.random.default_rng(0)
    print(f"[{a.name}] count", flush=True)
    main_c, main_d = count_metrics(count_run(a.run, a.weights, raw("count")), a.boot, rng)
    out["count"] = main_c
    for tag in ("dark", "small"):
        print(f"[{a.name}] count {tag}", flush=True)
        c, d = count_metrics(count_run(a.run, a.weights, raw(f"count_{tag}"), "--test-data", f"{PROBE}_{tag}"), a.boot, rng)
        drop = [x["box"] - y["box"] for x, y in zip(main_d, d)]       # independent test sets, independent draws
        out[f"count_{tag}"] = {"box": c["box"], "box_ci": c["box_ci"], "total": c["total"], "total_ci": c["total_ci"],
                               "box_drop": main_c["box"] - c["box"], "box_drop_ci": ci(drop), "n_test": c["n_test"]}
    print(f"[{a.name}] extrapolation", flush=True)
    e = count_run(a.run, a.weights, raw("extrap"), "--readout", "linear-mean", "--train-max", "2", "4")
    out["extrap"] = {f"{t}_n{r_['n']}_mean_pred": r_["mean_pred"] for t, lo in (("box", 3), ("total", 5))
                     for r_ in e[f"linear-mean/{t}"] if r_["n"] >= lo}    # counts beyond the readout's fit range
    out["voe"] = voe(a.run, step)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(a.out_dir, f"{a.name}.json"), "w"), indent=1)
    o, c = out["object"][str(LAYER)], out["count"]
    fmt = lambda v, k, s=1: f"{v[k] * s:.1f} [{v[k + '_ci'][0] * s:.1f}, {v[k + '_ci'][1] * s:.1f}]"  # noqa: E731
    print(f"[{a.name}] step {step}: AUC {o['grouping']:.3f} [{o['grouping_ci'][0]:.3f}, {o['grouping_ci'][1]:.3f}]  "
          f"moved id {fmt(o, 'identity_moved', 100)}  box {fmt(c, 'box')}  area-size {fmt(c, 'area_minus_size')}  "
          f"dark drop {fmt(out['count_dark'], 'box_drop')}  small drop {fmt(out['count_small'], 'box_drop')}  "
          f"size R2 {o['size_r2']:.2f}  depth R2 {o['depth3d_r2']:.2f}")


if __name__ == "__main__":
    main()
