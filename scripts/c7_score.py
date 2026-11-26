"""C7: pre-registered confirmation of C6's graded signal on fresh items (docs/2026-11-27-c7-edge-confirmation.md).

Same blocks as C6 (one source's 7 histories x one donor's 1/2/3-object endings, area-matched, hopper masked,
c6_items / c6_manifest). Per history h, the graded preference p(h) = S[h, 1-obj] - S[h, 3-obj] (> 0: the
prediction is closer to the 3-object ending). A constant donor bias cancels in every edge below.

Edges (pairs of histories differing in exactly ONE factor; correct when p moves the way the count does; ties
within 1e-12 get half credit; edges averaged equally within a factor, then blocks averaged):
  N       (N1-,N2-) (N1A,N2A) (N1AR,N2AR)      correct: p(N=2) > p(N=1)
  add     (N1-,N1A) (N2-,N2A) (N2R,N2AR)       correct: p(with add) > p(without)
  remove  (N1A,N1AR) (N2-,N2R) (N2A,N2AR)      correct: p(without remove) > p(with)
  conflict (N1A [final 2], N2R [final 1])      correct: p(N1A) > p(N2R)  (initial-count cue points the other way)
A reader that lacks a factor ties on that factor's edges (exactly 0.50). Readers: state counter (p = final),
symbolic single-feature readers (exact edge ceilings), the C6 shortcut readers (meta_area, meta_volume,
pixel_last_ctx, pixel_first; p from their distance matrices, all three directions via sign), untrained.
Bootstrap: 10,000 whole-block resamples stratified by slot order.

    python scripts/c7_score.py --manifest M.json --out OUT.json --run R --weights W --name N [...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from c6_items import CELLS  # noqa: E402
from c6_score import (FIN, actual_finals, features, load_model, mask_hopper, meta_area_matrix,  # noqa: E402
                      meta_volume_matrix, model_matrix, pixel_matrix, block_endpoints)

NAMES = [c[0] for c in CELLS]
IX = {n: i for i, n in enumerate(NAMES)}
EDGES = {"N": [("N1-", "N2-"), ("N1A", "N2A"), ("N1AR", "N2AR")],
         "add": [("N1-", "N1A"), ("N2-", "N2A"), ("N2R", "N2AR")],
         "remove": [("N1AR", "N1A"), ("N2R", "N2-"), ("N2AR", "N2A")],   # (lower count, higher count)
         "conflict": [("N2R", "N1A")]}
TIE = 1e-12
for k, es in EDGES.items():       # every edge is (lower final, higher final)
    for lo, hi in es:
        assert FIN[IX[lo]] < FIN[IX[hi]], (k, lo, hi)


def edge_scores(p: np.ndarray) -> dict[str, float]:
    out = {}
    for k, es in EDGES.items():
        v = [0.5 if abs(p[IX[hi]] - p[IX[lo]]) <= TIE else float(p[IX[hi]] > p[IX[lo]]) for lo, hi in es]
        out[k] = float(np.mean(v))
    out["mean_3"] = (out["N"] + out["add"] + out["remove"]) / 3
    out["min_3"] = min(out["N"], out["add"], out["remove"])
    return out


def pref(S: np.ndarray) -> np.ndarray:
    return S[:, 0] - S[:, 2]


def symbolic_edge_ceilings() -> dict:
    """Best edge accuracy for a reader that sees only one feature: 1 where the feature differs on an edge
    (best orientation), 0.5 where it is equal. Per slot order."""
    out = {}
    for af in (True, False):
        f = features(af)
        out[str(af)] = {nm: {k: float(np.mean([1.0 if v[IX[lo]] != v[IX[hi]] else 0.5 for lo, hi in es]))
                             for k, es in EDGES.items()} for nm, v in f.items()}
    return out


def bootstrap(per: dict[str, list[dict]], strata: np.ndarray, n: int, pairs: list, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    keys = list(next(iter(per.values()))[0])
    arr = {nm: np.array([[b[k] for k in keys] for b in bl]) for nm, bl in per.items()}
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    draws = {nm: [] for nm in arr}
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        for nm, A in arr.items():
            draws[nm].append(A[idx].mean(0))
    out = {nm: {k: {"est": float(A[:, j].mean()), "ci": np.percentile(np.array(draws[nm])[:, j], [2.5, 97.5]).tolist()}
                for j, k in enumerate(keys)} for nm, A in arr.items()}
    out["paired"] = {}
    for a, b in pairs:
        D = np.array(draws[a]) - np.array(draws[b])
        out["paired"][f"{a} - {b}"] = {k: {"est": float((arr[a] - arr[b])[:, j].mean()),
                                           "ci": np.percentile(D[:, j], [2.5, 97.5]).tolist()} for j, k in enumerate(keys)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--weights", action="append", default=[])
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--boot", type=int, default=10_000)
    a = ap.parse_args()
    if not (len(a.run) == len(a.weights) == len(a.name)):
        raise SystemExit("--run/--weights/--name must pair up")
    man = json.load(open(a.manifest))
    for d_, h in man["items_sha256"].items():
        if hashlib.sha256(open(os.path.join(d_, "items.jsonl"), "rb").read()).hexdigest() != h:
            raise SystemExit(f"{d_}/items.jsonl changed since the manifest was frozen")
    srec = {json.loads(l)["seed"]: json.loads(l) for l in open(os.path.join(man["sources_dir"], "items.jsonl"))}
    drec = {json.loads(l)["seed"]: json.loads(l) for dd in man["donors_dirs"] for l in open(os.path.join(dd, "items.jsonl"))}
    blocks = []
    for p in man["pairs"]:
        sp = os.path.join(man["sources_dir"], f"source_{p['source']:09d}.npz")
        dp = os.path.join(p["donor_dir"], f"donor_{p['donor']:09d}.npz")
        for path, h in ((sp, p["source_sha256"]), (dp, p["donor_sha256"])):
            if hashlib.sha256(open(path, "rb").read()).hexdigest() != h:
                raise SystemExit(f"{path} changed since the manifest was frozen")
        s, d = srec[p["source"]], drec[p["donor"]]
        if s["names"] != NAMES or [int(x) for x in d["names"]] != [1, 2, 3] or actual_finals(s).tolist() != FIN.tolist():
            raise SystemExit(f"block {p['source']}/{p['donor']}: order or simulated finals wrong")
        blocks.append((p, s, d, sp, dp))
    strata = np.array([s["add_first"] for _, s, _, _, _ in blocks])
    res = {"manifest": a.manifest, "n_blocks": len(blocks),
           "add_first": {str(k): int((strata == k).sum()) for k in (True, False)},
           "symbolic_edge_ceilings": symbolic_edge_ceilings(), "models": {}}
    per = {"state_counter": [edge_scores(actual_finals(s).astype(float)) for _, s, _, _, _ in blocks]}
    readers = {f"{r}:{d_}": [] for r in ("meta_area", "meta_volume", "pixel_last_ctx", "pixel_first") for d_ in ("pos", "neg")}
    readers["pixel_area:pos"] = []
    for p, s, d, sp, dp in blocks:
        zs, zd = np.load(sp), np.load(dp)
        xs = mask_hopper(zs["frames"], zs["hopper_px"], zs["hopper_ring"], s["ctx"])
        common = np.array(p["common"])
        mats = {"meta_area": meta_area_matrix(s, d), "meta_volume": meta_volume_matrix(s, d),
                "pixel_last_ctx": pixel_matrix(xs, zd["frames"], s["ctx"], s["first_post_slot"], common, "last_ctx"),
                "pixel_first": pixel_matrix(xs, zd["frames"], s["ctx"], s["first_post_slot"], common, "first")}
        for r, M in mats.items():   # 'pos': nearest-distance preference; 'neg': its reversal (farthest / inverted)
            readers[f"{r}:pos"].append(edge_scores(pref(M)))
            readers[f"{r}:neg"].append(edge_scores(-pref(M)))
        # visible-area reader: any count-monotone source matched on the donor's PIXEL area (common patches) gets an
        # edge right exactly when the 3-object ending covers more pixels; p = final x log(pix3 / pix1)
        cov = zd["coverage"].reshape(3, -1)[:, common].sum(1)
        readers["pixel_area:pos"].append(edge_scores(actual_finals(s) * np.log(cov[2] / cov[0])))
    per.update(readers)
    torch.set_num_threads(16)
    for run, w, name in zip(a.run, a.weights, a.name):
        enc, tgt, pred, step = load_model(run, w)
        enc.eval(); tgt.eval(); pred.eval()
        Ms = []
        for p, s, d, sp, dp in blocks:
            zs, zd = np.load(sp), np.load(dp)
            xs = mask_hopper(zs["frames"], zs["hopper_px"], zs["hopper_ring"], s["ctx"])
            Ms.append(model_matrix(xs, zd["frames"], s["ctx"], s["first_post_slot"], np.array(p["common"]), enc, tgt, pred))
        per[name] = [edge_scores(pref(S)) for S in Ms]
        c6 = [block_endpoints(S) for S in Ms]
        res["models"][name] = {"run": run, "weights": w, "step": step,
                               "c6_endpoints": {k: float(np.mean([b[k] for b in c6])) for k in c6[0]}}
        np.savez_compressed(os.path.splitext(a.out)[0] + f"_{name}_mats.npz", mats=np.stack(Ms),
                            source=np.array([p["source"] for p, *_ in blocks]), donor=np.array([p["donor"] for p, *_ in blocks]))
        print(f"[{name}] " + "  ".join(f"{k} {np.mean([b[k] for b in per[name]]):.3f}" for k in per[name][0]), flush=True)
    pairs = [(m, r) for m in res["models"] for r in readers] + [(m1, m2) for m1 in res["models"] for m2 in res["models"] if m1 != m2]
    res["bootstrap"] = bootstrap(per, strata, a.boot, pairs)
    res["by_order"] = {nm: {str(o): {k: float(np.mean([b[k] for b, st_ in zip(bl, strata) if st_ == o])) for k in bl[0]}
                            for o in (True, False)} for nm, bl in per.items()}
    for nm in per:
        if nm not in res["models"]:
            print(f"[reader] {nm:22s} " + "  ".join(f"{k} {res['bootstrap'][nm][k]['est']:.3f}" for k in ("N", "add", "remove", "conflict")), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
