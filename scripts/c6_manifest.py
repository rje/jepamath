"""C6 pairing + freeze: one donor per source, WITHOUT replacement, from geometry and timing only.

A source and a donor are compatible when ctx, first post-reveal slot and clip length are identical, box-token
IoU >= --min-iou, the common token mask is non-empty, and the donor's three endings are area-matched ON THAT
PAIR'S COMMON PATCHES (largest / smallest summed post-reveal object coverage <= --max-area-ratio). The
assignment maximises total IoU (scipy linear_sum_assignment). Several donor directories (the registered pool
plus any registered top-up) are pooled.

--preview prints the counts and writes nothing. Without it, the manifest (pairs, common masks, seed windows,
generator rejection totals, sha256 of every NPZ and items.jsonl) is written once (refuses to overwrite), and
the freeze is REFUSED below --min-pairs: render the registered top-up first. Freeze before any checkpoint
inference.

    python scripts/c6_manifest.py --sources DIR --donors DIR [--donors DIR2] --out MANIFEST.json [--preview]
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
        path = os.path.join(d, f"{role}_{r['seed']:09d}.npz")
        z = np.load(path)
        out.append({"seed": r["seed"], "ctx": r["ctx"], "first": r["first_post_slot"], "T": r["n_frames"],
                    "box": z["box_tokens"].astype(bool), "cov": z["coverage"] if "coverage" in z else None,
                    "add_first": r.get("add_first"), "path": path, "dir": d, "meta31": meta31(r) if role == "donor" else None})
    return out


def meta31(r: dict) -> float:
    """log(metadata area of the 3-object ending / the 1-object ending), sum of pi r^2 of the box objects."""
    import math

    def rad(o):
        return o["size"][0] / 0.8 if o["shape"] == "box" else o["size"][0]
    a = [sum(math.pi * rad(o) ** 2 for o in r["episodes"][n]["objects"] if o["start_region"] == "box") for n in r["names"]]
    return math.log(a[2] / a[0])


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rejects(d: str) -> int:
    p = os.path.join(d, "rejects.jsonl")
    return sum(1 for _ in open(p)) if os.path.exists(p) else 0


def main():
    from scipy.optimize import linear_sum_assignment
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True)
    ap.add_argument("--donors", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-iou", type=float, default=0.6)
    ap.add_argument("--max-area-ratio", type=float, default=1.2)
    ap.add_argument("--min-pairs", type=int, default=360)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--balance-area", action="store_true",
                    help="C8: within each slot order, equal numbers of pairs with the 3-object ending larger / smaller "
                         "in metadata area AND in pixel area on the pair's common patches")
    ap.add_argument("--per-order", type=int, default=0,
                    help="keep exactly this many pairs per slot order (lowest source seeds first); 0 = all")
    a = ap.parse_args()
    if not a.preview and os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists: the manifest is frozen once written")
    S = load(a.sources, "source")
    D = [d for dd in a.donors for d in load(dd, "donor")]
    if len({x["seed"] for x in S}) != len(S) or len({x["seed"] for x in D}) != len(D):
        raise SystemExit("duplicate seeds")
    W = np.full((len(S), len(D)), -1.0)
    area_fail = 0
    for i, s in enumerate(S):
        for j, d in enumerate(D):
            if (s["ctx"], s["first"], s["T"]) != (d["ctx"], d["first"], d["T"]):
                continue
            common = s["box"] & d["box"]
            iou = common.sum() / max(1, (s["box"] | d["box"]).sum())
            if iou < a.min_iou or common.sum() == 0:
                continue
            ar = d["cov"][:, common].sum(1)
            if ar.max() / max(ar.min(), 1e-9) > a.max_area_ratio:
                area_fail += 1
                continue
            W[i, j] = iou
    r, c = linear_sum_assignment(np.where(W >= 0, W, -1e3), maximize=True)
    pairs = [(i, j) for i, j in zip(r, c) if W[i, j] >= 0]
    def signs(i, j):
        common = S[i]["box"] & D[j]["box"]
        pix = D[j]["cov"][:, common].sum(1)
        if abs(pix[2] - pix[0]) <= 1e-9 or abs(D[j]["meta31"]) <= 1e-12:
            return None             # an exact tie belongs to neither sign: such pairs are excluded from balancing
        return (bool(D[j]["meta31"] > 0), bool(pix[2] > pix[0]))
    if a.balance_area:
        # Amendment 2 (27 Nov, before freeze): cell quotas are built INTO the assignment. For each slot order and a
        # discordant quota b (largest feasible, at most per_order/4), the cells are filled scarcest-first, each by a
        # max-IoU assignment over the still-unused sources and donors whose pair falls in that cell; the two
        # concordant cells get per_order/2 - b each. Donors are used once across both orders.
        if not a.per_order or a.per_order % 2:
            raise SystemExit("--balance-area needs an even --per-order")
        used_d, kept = set(), []
        order_cells = [(False, True), (True, False), (False, False), (True, True)]
        for o in (True, False):
            src_o = [i for i in range(len(S)) if S[i]["add_first"] == o]
            sign = {(i, j): signs(i, j) for i in src_o for j in range(len(D)) if W[i, j] >= 0}
            done = None
            for b in range(a.per_order // 4, -1, -1):
                quota = {(False, True): b, (True, False): b, (False, False): a.per_order // 2 - b,
                         (True, True): a.per_order // 2 - b}
                us, ud, got = set(), set(used_d), []
                ok = True
                for cell in order_cells:
                    rows = [i for i in src_o if i not in us]
                    cols = [j for j in range(len(D)) if j not in ud]
                    M = np.array([[W[i, j] if W[i, j] >= 0 and sign.get((i, j)) == cell else -1.0 for j in cols]
                                  for i in rows]) if rows and cols else np.zeros((0, 0))
                    if M.size == 0:
                        ok = quota[cell] == 0
                        if not ok:
                            break
                        continue
                    rr, cc = linear_sum_assignment(np.where(M >= 0, M, -1e3), maximize=True)
                    cand = sorted([(M[x, y], rows[x], cols[y]) for x, y in zip(rr, cc) if M[x, y] >= 0], reverse=True)
                    if len(cand) < quota[cell]:
                        ok = False
                        break
                    for _, i, j in cand[:quota[cell]]:
                        us.add(i); ud.add(j); got.append((i, j))
                if ok:
                    done = (b, got, ud)
                    break
            if done is None:
                raise SystemExit(f"slot order {o}: no balanced assignment of {a.per_order}: render the registered top-up")
            b, got, ud = done
            used_d = ud
            kept += got
            print(f"slot order {o}: balanced assignment with discordant {b}+{b}, concordant "
                  f"{a.per_order // 2 - b}+{a.per_order // 2 - b}")
        pairs = kept
    elif a.per_order:
        kept = []
        for o in (True, False):
            po = sorted([(i, j) for i, j in pairs if S[i]["add_first"] == o], key=lambda t: S[t[0]]["seed"])
            if len(po) < a.per_order:
                raise SystemExit(f"slot order {o}: only {len(po)} pairs < {a.per_order}: render the registered top-up")
            kept += po[:a.per_order]
        pairs = kept
    used = {i for i, _ in pairs}
    summary = {"n_sources": len(S), "n_donors": len(D), "n_pairs": len(pairs),
               "compatible_pairs_failing_area": area_fail,
               "add_first_counts": {str(k): sum(S[i]["add_first"] == k for i, _ in pairs) for k in (True, False)},
               "iou_mean": float(np.mean([W[i, j] for i, j in pairs])) if pairs else None,
               "source_seed_window": [min(x["seed"] for x in S), max(x["seed"] for x in S)],
               "donor_seed_window": [min(x["seed"] for x in D), max(x["seed"] for x in D)],
               "generator_rejects": {a.sources: rejects(a.sources), **{dd: rejects(dd) for dd in a.donors}}}
    print(json.dumps(summary, indent=1))
    if a.preview:
        return
    if len(pairs) < a.min_pairs:
        raise SystemExit(f"only {len(pairs)} pairs < {a.min_pairs}: render the registered donor top-up first")
    man = {"sources_dir": a.sources, "donors_dirs": a.donors, "min_iou": a.min_iou, "balance_area": a.balance_area,
           "max_area_ratio": a.max_area_ratio, **summary,
           "items_sha256": {d: sha(os.path.join(d, "items.jsonl")) for d in [a.sources, *a.donors]},
           "sources_unmatched": [S[i]["seed"] for i in range(len(S)) if i not in used],
           "pairs": [{"source": S[i]["seed"], "donor": D[j]["seed"], "donor_dir": D[j]["dir"], "iou": float(W[i, j]),
                      "common": np.flatnonzero((S[i]["box"] & D[j]["box"]).ravel()).tolist(),
                      "source_sha256": sha(S[i]["path"]), "donor_sha256": sha(D[j]["path"])} for i, j in pairs]}
    json.dump(man, open(a.out, "x"), indent=1)


if __name__ == "__main__":
    main()
