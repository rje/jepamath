"""C10 scoring (docs/2026-11-27-c10-count-vs-route-size.md): the frozen P1 ridge probe (fit on C8) on C10's 17-history
scenes, scored on within-scene edges where the route (E1, E2), the start count (E3) or the carried size (E4) disagrees
with the final count. Readers built from the simulation check that each endpoint defeats what it should.

    CUDA_VISIBLE_DEVICES= python scripts/c10_score.py --items DIR --out OUT.json --cache CACHE_DIR \
        --run R --weights W --name N [...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from c6_score import load_model, mask_hopper  # noqa: E402
from c9_score import SCALARS, final_objects  # noqa: E402
from p1_count_probe import c8_items, feats, ridge_cv  # noqa: E402
from p2_controls import model_feats  # noqa: E402

C8 = "/mnt/data/jepamath/c8_v1/manifest.json"
TIE = 1e-12
ENDPOINTS = ("E1_same_route", "E2_route_opposed", "E3_start_opposed", "E4_size_opposed")


def c8_features(name, run, w, enc, tgt, pred, cache):
    p = os.path.join(cache, f"c8_pred_{name}.npz")
    if os.path.exists(p):
        z = np.load(p)
        return z["X"], z["y"], z["g"]
    fm = json.load(open(C8))
    X, y, g = [], [], []
    for seed, s, z, fin in c8_items(fm):
        m = mask_hopper(z["frames"], z["hopper_px"], z["hopper_ring"], s["ctx"])
        P, *_ = feats(z["frames"], m, s["ctx"], s["first_post_slot"], z["box_tokens"].astype(bool), enc, tgt, pred)
        X.append(P); y.append(fin); g += [seed] * len(fin)
    X, y, g = np.concatenate(X), np.concatenate(y), np.array(g)
    np.savez(p, X=X, y=y, g=g, run=run, weights=w)
    return X, y, g


def scene_info(rec):
    out = {}
    for n in rec["names"]:
        fo = final_objects(rec, n)
        m = rec["meta"][n]
        out[n] = {"final": len(fo), "n0": m["n0"], "net": m["net"], "route": m["route_name"],
                  **{k: SCALARS[k](fo) for k in ("radius_sum", "area", "volume")}}
    return out


def edges(info: dict, c: dict) -> dict:
    """Per-scene endpoint means (NaN when a scene has no edge of that kind) + E5 parts."""
    acc = {e: [] for e in ENDPOINTS}
    eq, gap1 = [], []
    names = list(info)
    for a in names:
        for b in names:
            A, Bb = info[a], info[b]
            if A["final"] == Bb["final"] and a < b and A["route"] != Bb["route"]:
                eq.append(abs(c[a] - c[b]))
            if A["final"] <= Bb["final"]:
                continue
            if A["final"] - Bb["final"] == 1:
                gap1.append(c[a] - c[b])
            v = 0.5 if abs(c[a] - c[b]) <= TIE else float(c[a] > c[b])
            if A["route"] == Bb["route"]:
                acc["E1_same_route"].append(v)
            if A["net"] < Bb["net"]:
                acc["E2_route_opposed"].append(v)
            if A["n0"] < Bb["n0"]:
                acc["E3_start_opposed"].append(v)
            if all(A[k] < Bb[k] for k in ("radius_sum", "area", "volume")):
                acc["E4_size_opposed"].append(v)
    row = {e: (float(np.mean(v)) if v else float("nan")) for e, v in acc.items()}
    row["_n"] = {e: len(v) for e, v in acc.items()}
    row["_eq"], row["_gap1"] = (float(np.mean(eq)) if eq else float("nan")), (float(np.mean(gap1)) if gap1 else float("nan"))
    return row


def bootstrap(per: dict, strata: np.ndarray, n: int, pairs, seed=0):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    arr = {m: np.array([[r[e] for e in ENDPOINTS] for r in rows]) for m, rows in per.items()}
    draws = {m: [] for m in arr}
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        for m, A in arr.items():
            draws[m].append(np.nanmean(A[idx], 0))
    out = {m: {e: {"est": float(np.nanmean(A[:, j])), "ci": np.percentile(np.array(draws[m])[:, j], [2.5, 97.5]).tolist()}
               for j, e in enumerate(ENDPOINTS)} for m, A in arr.items()}
    out["paired"] = {}
    for a_, b_ in pairs:
        D = np.array(draws[a_]) - np.array(draws[b_])
        out["paired"][f"{a_} - {b_}"] = {e: {"est": float(np.nanmean(arr[a_][:, j]) - np.nanmean(arr[b_][:, j])),
                                             "ci": np.percentile(D[:, j], [2.5, 97.5]).tolist()} for j, e in enumerate(ENDPOINTS)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--weights", action="append", default=[])
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--limit", type=int, default=0, help="smoke test only")
    a = ap.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("run with CUDA_VISIBLE_DEVICES=")
    if not (len(a.run) == len(a.weights) == len(a.name)):
        raise SystemExit("--run/--weights/--name must pair up")
    os.makedirs(a.cache, exist_ok=True)
    recs = [json.loads(l) for l in open(os.path.join(a.items, "items.jsonl"))]
    if a.limit:
        recs = recs[: a.limit]
    infos = [scene_info(r) for r in recs]
    strata = np.array([r["meta"][r["names"][0]]["shape"] for r in recs])
    for r, inf in zip(recs, infos):
        bad = [n for n in r["names"] if inf[n]["final"] != r["meta"][n]["final"]]
        if bad:
            raise SystemExit(f"scene {r['seed']}: simulated final differs from the schedule for {bad}")
    per, res = {}, {"items": a.items, "n_scenes": len(recs), "models": {}}
    # readers from the simulation (validity)
    readers = {"state": lambda i: i["final"], "net": lambda i: i["net"], "start": lambda i: i["n0"],
               "radius_sum": lambda i: i["radius_sum"], "area": lambda i: i["area"], "volume": lambda i: i["volume"]}
    for rn, f in readers.items():
        per[f"reader:{rn}"] = [edges(inf, {n: float(f(inf[n])) for n in inf}) for inf in infos]
    res["edge_counts"] = {e: int(sum(r["_n"][e] for r in per["reader:state"])) for e in ENDPOINTS}
    res["scenes_with_E4"] = int(sum(r["_n"]["E4_size_opposed"] > 0 for r in per["reader:state"]))
    torch.set_num_threads(16)
    for run, w, name in zip(a.run, a.weights, a.name):
        enc, tgt, pred, step = load_model(run, w)
        enc.eval(); tgt.eval(); pred.eval()
        X, y, g = c8_features(name, run, w, enc, tgt, pred, a.cache)
        probe, info = ridge_cv(X, y, g)
        rows, cs = [], []
        for r, inf in zip(recs, infos):
            z = np.load(os.path.join(a.items, f"source_{r['seed']:09d}.npz"))
            m = mask_hopper(z["frames"], z["hopper_px"], z["hopper_ring"], r["ctx"])
            _, _, pp, _ = model_feats(m, r["ctx"], r["first_post_slot"], z["box_tokens"].astype(bool), enc, pred)
            c = dict(zip(r["names"], map(float, probe.predict(pp))))
            rows.append(edges(inf, c)); cs.append(c)
        per[name] = rows
        e5 = float(np.nanmean([x["_eq"] for x in rows]) / np.nanmean([x["_gap1"] for x in rows]))
        by_final = {int(k): float(np.mean([c[n] for c, inf in zip(cs, infos) for n in inf if inf[n]["final"] == k]))
                    for k in (1, 2, 3)}
        res["models"][name] = {"run": run, "weights": w, "step": step, "fit": info, "E5_equal_count_spread": e5,
                               "mean_c_by_final": by_final}
        print(f"[{name}] " + "  ".join(f"{e} {np.nanmean([x[e] for x in rows]):.3f}" for e in ENDPOINTS)
              + f"  E5 {e5:.3f}  c|final {by_final}", flush=True)
    for rn in readers:
        rows = per[f"reader:{rn}"]
        print(f"[reader:{rn}] " + "  ".join(f"{e} {np.nanmean([x[e] for x in rows]):.3f}" for e in ENDPOINTS), flush=True)
    pairs = [(m, "untrained") for m in res["models"] if m != "untrained" and "untrained" in res["models"]]
    res["bootstrap"] = bootstrap(per, strata, a.boot, pairs)
    res["edge_counts"]["note"] = "ordered pairs with final(a) > final(b), summed over scenes"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
