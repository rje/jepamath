"""C6 pairing + freeze: one donor per source, WITHOUT replacement, from geometry and timing only.

A source and a donor are compatible when ctx, first post-reveal slot and clip length are identical, box-token
IoU >= --min-iou and the common token mask is non-empty. The donor area gate (largest / smallest measured
post-reveal box-object area <= --max-area-ratio) is applied first. The assignment maximises total IoU
(scipy linear_sum_assignment); unmatched sources are dropped and counted. The manifest (pairs, common masks
by hash, seeds, rejection counts, file sha256s) is written once and refuses to overwrite: it is frozen before
any checkpoint inference.

    python scripts/c6_manifest.py --sources DIR --donors DIR --out MANIFEST.json [--min-iou 0.6 --max-area-ratio 1.2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np


def load(d: str, role: str) -> list[dict]:
    out = []
    for line in open(os.path.join(d, "items.jsonl")):
        r = json.loads(line)
        z = np.load(os.path.join(d, f"{role}_{r['seed']:09d}.npz"))
        out.append({"seed": r["seed"], "ctx": r["ctx"], "first": r["first_post_slot"], "T": r["n_frames"],
                    "box": z["box_tokens"].astype(bool), "area_ratio": r.get("area_ratio"),
                    "add_first": r.get("add_first"), "path": os.path.join(d, f"{role}_{r['seed']:09d}.npz")})
    if len({x["seed"] for x in out}) != len(out):
        raise SystemExit(f"duplicate seeds in {d}")
    return out


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    from scipy.optimize import linear_sum_assignment
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True)
    ap.add_argument("--donors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-iou", type=float, default=0.6)
    ap.add_argument("--max-area-ratio", type=float, default=1.2)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists: the manifest is frozen once written")
    S, D0 = load(a.sources, "source"), load(a.donors, "donor")
    D = [d for d in D0 if d["area_ratio"] <= a.max_area_ratio]
    W = np.full((len(S), len(D)), -1.0)
    for i, s in enumerate(S):
        for j, d in enumerate(D):
            if (s["ctx"], s["first"], s["T"]) != (d["ctx"], d["first"], d["T"]):
                continue
            inter = (s["box"] & d["box"]).sum()
            iou = inter / max(1, (s["box"] | d["box"]).sum())
            if iou >= a.min_iou and inter > 0:
                W[i, j] = iou
    r, c = linear_sum_assignment(np.where(W >= 0, W, -1e3), maximize=True)
    pairs = [(i, j) for i, j in zip(r, c) if W[i, j] >= 0]
    used = {i for i, _ in pairs}
    man = {"sources_dir": a.sources, "donors_dir": a.donors, "min_iou": a.min_iou, "max_area_ratio": a.max_area_ratio,
           "n_sources": len(S), "n_donors": len(D0), "donors_area_rejected": len(D0) - len(D),
           "n_pairs": len(pairs), "sources_unmatched": [S[i]["seed"] for i in range(len(S)) if i not in used],
           "add_first_counts": {str(k): sum(S[i]["add_first"] == k for i, _ in pairs) for k in (True, False)},
           "iou_mean": float(np.mean([W[i, j] for i, j in pairs])) if pairs else None,
           "pairs": [{"source": S[i]["seed"], "donor": D[j]["seed"], "iou": float(W[i, j]),
                      "common": np.flatnonzero((S[i]["box"] & D[j]["box"]).ravel()).tolist(),
                      "source_sha256": sha(S[i]["path"]), "donor_sha256": sha(D[j]["path"])} for i, j in pairs]}
    json.dump(man, open(a.out, "x"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "pairs"}, indent=1))


if __name__ == "__main__":
    main()
