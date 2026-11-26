"""C9 pairing + scoring (DRAFT design docs/2026-11-27-c9-number-vs-area-draft.md; registered in the C9 pre-registration).

A block = one source scene (12 clips: families orig / area / volume x histories H, L-N, L-add, L-remove; c9_items)
+ one 'rev' donor (3-object ending smaller, more compact, less area / volume / radius) + one 'fwd' donor (all
reversed), each with endings of 2 and 3 objects. Pairing (--freeze): per layout, max-IoU assignment with unique
donors among compatible pairs (same ctx / first post slot / length, box IoU >= 0.6); a block needs both. The common
mask is per (source, donor).

Score: p(h, donor) = d(h, 2-object ending) - d(h, 3-object ending) (> 0: closer to the 3-object ending). Edge L -> H
is correct when p(H) > p(L) (H has 3 objects, every L has 2); ties within 1e-12 half credit. Each edge is averaged
over the two donors equally.
  conflict:<fam>  L-remove -> H     (H gains an object but, in 'orig', LESS carried area and volume)
  LN:<fam>        L-N -> H
  Ladd:<fam>      L-add -> H
  pair_area:<edge>    mean(<edge>:orig, <edge>:area)     exact 0.50 for ANY reader p = f(final summed area, donor)
  pair_volume:<edge>  mean(<edge>:orig, <edge>:volume)   exact 0.50 for ANY reader p = f(final summed volume, donor)
Readers: state counter; scalar source readers (final summed area / volume / radius) x donor contrasts (metadata
area / volume / max radius / pixel area / spread), both polarities; first-frame / last-context pixel readers.
Bootstrap: 10,000 whole-block resamples stratified by shape.

    python scripts/c9_score.py --sources S --rev R --fwd F --manifest M.json --freeze
    python scripts/c9_score.py --manifest M.json --out OUT.json --run R --weights W --name N [...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from c6_score import load_model, mask_hopper, model_matrix, pixel_matrix  # noqa: E402
from c9_items import FAMILIES, HISTORIES  # noqa: E402

TIE = 1e-12
EDGES = {"conflict": "L-remove", "LN": "L-N", "Ladd": "L-add"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load(d, role):
    out = []
    for line in open(os.path.join(d, "items.jsonl")):
        r = json.loads(line)
        path = os.path.join(d, f"{role}_{r['seed']:09d}.npz")
        out.append({"rec": r, "path": path, "box": np.load(path)["box_tokens"].astype(bool)})
    return out


def freeze(a):
    from scipy.optimize import linear_sum_assignment
    if os.path.exists(a.manifest):
        raise SystemExit(f"{a.manifest} exists: frozen")
    S, R, F = load(a.sources, "source"), load(a.rev, "donor"), load(a.fwd, "donor")

    def assign(D):
        W = np.full((len(S), len(D)), -1.0)
        for i, s in enumerate(S):
            for j, d in enumerate(D):
                if (s["rec"]["ctx"], s["rec"]["first_post_slot"], s["rec"]["n_frames"]) != \
                        (d["rec"]["ctx"], d["rec"]["first_post_slot"], d["rec"]["n_frames"]):
                    continue
                inter = (s["box"] & d["box"]).sum()
                iou = inter / max(1, (s["box"] | d["box"]).sum())
                if iou >= a.min_iou and inter > 0:
                    W[i, j] = iou
        r, c = linear_sum_assignment(np.where(W >= 0, W, -1e3), maximize=True)
        return {i: (j, W[i, j]) for i, j in zip(r, c) if W[i, j] >= 0}
    mr, mf = assign(R), assign(F)
    blocks = [i for i in range(len(S)) if i in mr and i in mf]
    if a.per_shape:
        keep = []
        for sh in ("sphere", "box", "cylinder"):
            bs = sorted([i for i in blocks if S[i]["rec"]["meta"]["orig:H"]["shape"] == sh], key=lambda i: S[i]["rec"]["seed"])
            if len(bs) < a.per_shape:
                raise SystemExit(f"shape {sh}: only {len(bs)} complete blocks < {a.per_shape}")
            keep += bs[:a.per_shape]
        blocks = keep
    man = {"sources_dir": a.sources, "rev_dir": a.rev, "fwd_dir": a.fwd, "min_iou": a.min_iou,
           "items_sha256": {d: sha(os.path.join(d, "items.jsonl")) for d in (a.sources, a.rev, a.fwd)},
           "n_blocks": len(blocks), "blocks": []}
    for i in blocks:
        b = {"source": S[i]["rec"]["seed"], "source_sha256": sha(S[i]["path"]), "shape": S[i]["rec"]["meta"]["orig:H"]["shape"]}
        for lay, D, m in (("rev", R, mr), ("fwd", F, mf)):
            j, iou = m[i]
            b[lay] = {"donor": D[j]["rec"]["seed"], "iou": float(iou), "sha256": sha(D[j]["path"]),
                      "common": np.flatnonzero((S[i]["box"] & D[j]["box"]).ravel()).tolist()}
        man["blocks"].append(b)
    json.dump(man, open(a.manifest, "x"), indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "blocks"}, indent=1))
    print("per shape:", {sh: sum(b["shape"] == sh for b in man["blocks"]) for sh in ("sphere", "box", "cylinder")})


# ---------------- quantities ----------------

def _r(o):
    return o["size"][0] / 0.8 if o["shape"] == "box" else o["size"][0]


def _vol(o):
    s = o["size"]
    if o["shape"] == "sphere":
        return 4 / 3 * math.pi * s[0] ** 3
    if o["shape"] == "box":
        return 8 * s[0] * s[1] * s[2]
    return 2 * math.pi * s[0] ** 2 * s[1]


def final_objects(rec, name):
    """Objects in the box at the end, from the simulated transfer records (not the schedule)."""
    ep = rec["episodes"][name]
    objs = {o["id"]: o for o in ep["objects"]}
    inbox = {o["id"] for o in ep["objects"] if o["start_region"] == "box"}
    for t in rec["transfers"][name]:
        if t["region_before"] == "box":
            inbox.discard(t["object"])
        if t["region_after"] == "box":
            inbox.add(t["object"])
    return [objs[i] for i in inbox]


SCALARS = {"count": lambda os_: float(len(os_)), "area": lambda os_: sum(math.pi * _r(o) ** 2 for o in os_),
           "volume": lambda os_: sum(_vol(o) for o in os_), "radius_sum": lambda os_: sum(_r(o) for o in os_)}


def donor_contrasts(drec, zd, common):
    """3-object minus 2-object ending contrasts of one donor (positive: the 3-ending has more)."""
    e = {int(k): v for k, v in zip(drec["names"], range(len(drec["names"])))}
    objs = {c: [o for o in drec["episodes"][str(c)]["objects"] if o["start_region"] == "box"] for c in (2, 3)}
    cov = zd["coverage"].reshape(2, -1)[:, common].sum(1)
    sp = drec["spread"]
    return {"meta_area": math.log(SCALARS["area"](objs[3]) / SCALARS["area"](objs[2])),
            "meta_volume": math.log(SCALARS["volume"](objs[3]) / SCALARS["volume"](objs[2])),
            "max_radius": max(_r(o) for o in objs[3]) - max(_r(o) for o in objs[2]),
            "pixel_area": math.log(cov[e[3]] / cov[e[2]]), "spread": sp["3"] - sp["2"]}


# ---------------- endpoints ----------------

def block_endpoints(P: dict[str, np.ndarray]) -> dict[str, float]:
    """P[layout] = p per source clip name (dict name -> value). Edge credit averaged over both donors."""
    out = {}
    for e, lo in EDGES.items():
        for fam in FAMILIES:
            v = []
            for lay in ("rev", "fwd"):
                hi_, lo_ = P[lay][f"{fam}:H"], P[lay][f"{fam}:{lo}"]
                v.append(0.5 if abs(hi_ - lo_) <= TIE else float(hi_ > lo_))
            out[f"{e}:{fam}"] = float(np.mean(v))
        out[f"pair_area:{e}"] = (out[f"{e}:orig"] + out[f"{e}:area"]) / 2
        out[f"pair_volume:{e}"] = (out[f"{e}:orig"] + out[f"{e}:volume"]) / 2
    return out


def bootstrap(per, strata, n, pairs, seed=0):
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
    for a_, b_ in pairs:
        D = np.array(draws[a_]) - np.array(draws[b_])
        out["paired"][f"{a_} - {b_}"] = {k: {"est": float((arr[a_] - arr[b_])[:, j].mean()),
                                             "ci": np.percentile(D[:, j], [2.5, 97.5]).tolist()} for j, k in enumerate(keys)}
    return out


def score(a):
    man = json.load(open(a.manifest))
    for d_, h in man["items_sha256"].items():
        if sha(os.path.join(d_, "items.jsonl")) != h:
            raise SystemExit(f"{d_}/items.jsonl changed since the freeze")
    srec = {json.loads(l)["seed"]: json.loads(l) for l in open(os.path.join(man["sources_dir"], "items.jsonl"))}
    drec = {lay: {json.loads(l)["seed"]: json.loads(l) for l in open(os.path.join(man[f"{lay}_dir"], "items.jsonl"))}
            for lay in ("rev", "fwd")}
    blocks = []
    for b in man["blocks"]:
        sp = os.path.join(man["sources_dir"], f"source_{b['source']:09d}.npz")
        if sha(sp) != b["source_sha256"]:
            raise SystemExit(f"{sp} changed")
        dps = {}
        for lay in ("rev", "fwd"):
            dp = os.path.join(man[f"{lay}_dir"], f"donor_{b[lay]['donor']:09d}.npz")
            if sha(dp) != b[lay]["sha256"]:
                raise SystemExit(f"{dp} changed")
            dps[lay] = dp
        s = srec[b["source"]]
        fin = {n: len(final_objects(s, n)) for n in s["names"]}
        if any(fin[n] != (3 if n.endswith(":H") else 2) for n in s["names"]):
            raise SystemExit(f"source {b['source']}: simulated finals {fin}")
        blocks.append((b, s, {lay: drec[lay][b[lay]["donor"]] for lay in dps}, sp, dps))
    strata = np.array([b["shape"] for b, *_ in blocks])
    res = {"manifest": a.manifest, "n_blocks": len(blocks), "per_shape": {k: int((strata == k).sum()) for k in np.unique(strata)},
           "models": {}}
    per = {}
    # scalar readers: the source history's FINAL summed quantity q matched to each donor ending's q (nearest, or
    # farthest): p = |q_h - q_d2| - |q_h - q_d3|. 'count:nearest' is the state counter (must be 1.0). For area and
    # volume the paired endpoints are exactly 0.50 by construction (the companions swap the final values).
    for q in SCALARS:
        for rule in ("nearest", "farthest"):
            per[f"{q}:{rule}"] = []
    per["pixel_first"], per["pixel_last_ctx"] = [], []
    for b, s, dr, sp, dps in blocks:
        names = s["names"]
        finals = {n: final_objects(s, n) for n in names}
        zs = np.load(sp)
        xs = mask_hopper(zs["frames"], zs["hopper_px"], zs["hopper_ring"], s["ctx"])
        con = {lay: donor_contrasts(dr[lay], np.load(dps[lay]), np.array(b[lay]["common"])) for lay in dps}
        for q, f in SCALARS.items():
            dq = {lay: {c: f([o for o in dr[lay]["episodes"][str(c)]["objects"] if o["start_region"] == "box"])
                        for c in (2, 3)} for lay in dps}
            for rule, sg in (("nearest", 1), ("farthest", -1)):
                P = {lay: {n: sg * (abs(f(finals[n]) - dq[lay][2]) - abs(f(finals[n]) - dq[lay][3])) for n in names}
                     for lay in dps}
                per[f"{q}:{rule}"].append(block_endpoints(P))
        res.setdefault("donor_contrasts", []).append(con)
        for mode in ("first", "last_ctx"):
            P = {}
            for lay in dps:
                zd = np.load(dps[lay])
                M = pixel_matrix(xs, zd["frames"], s["ctx"], s["first_post_slot"], np.array(b[lay]["common"]), mode)
                P[lay] = {n: float(M[i, 0] - M[i, 1]) for i, n in enumerate(names)}
            per[f"pixel_{mode}"].append(block_endpoints(P))
    torch.set_num_threads(16)
    for run, w, name in zip(a.run, a.weights, a.name):
        enc, tgt, pred, step = load_model(run, w)
        enc.eval(); tgt.eval(); pred.eval()
        rows, mats = [], {}
        for b, s, dr, sp, dps in blocks:
            zs = np.load(sp)
            xs = mask_hopper(zs["frames"], zs["hopper_px"], zs["hopper_ring"], s["ctx"])
            P = {}
            for lay in dps:
                zd = np.load(dps[lay])
                M = model_matrix(xs, zd["frames"], s["ctx"], s["first_post_slot"], np.array(b[lay]["common"]), enc, tgt, pred)
                mats.setdefault(lay, []).append(M)
                P[lay] = {n: float(M[i, 0] - M[i, 1]) for i, n in enumerate(s["names"])}
            rows.append(block_endpoints(P))
        per[name] = rows
        res["models"][name] = {"run": run, "weights": w, "step": step}
        np.savez_compressed(os.path.splitext(a.out)[0] + f"_{name}_mats.npz", rev=np.stack(mats["rev"]), fwd=np.stack(mats["fwd"]),
                            source=np.array([b["source"] for b, *_ in blocks]))
        print(f"[{name}] " + "  ".join(f"{k} {np.mean([r[k] for r in rows]):.3f}" for k in rows[0]), flush=True)
    # the state counter: count reader (both donors, polarity +) is the reference; reported under its reader name
    pairs = [(m, "pixel_first") for m in res["models"]] + [(m, "pixel_last_ctx") for m in res["models"]] + \
            [(m1, m2) for m1 in res["models"] for m2 in res["models"] if m1 != m2]
    res["bootstrap"] = bootstrap(per, strata, a.boot, pairs)
    json.dump(res, open(a.out, "w"), indent=1)
    keys = ["conflict:orig", "conflict:area", "conflict:volume", "pair_area:conflict", "pair_volume:conflict"]
    for nm in per:
        if nm not in res["models"]:
            print(f"[reader] {nm:32s} " + "  ".join(f"{k} {res['bootstrap'][nm][k]['est']:.3f}" for k in keys), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--sources"); ap.add_argument("--rev"); ap.add_argument("--fwd")
    ap.add_argument("--min-iou", type=float, default=0.6)
    ap.add_argument("--per-shape", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--weights", action="append", default=[])
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--boot", type=int, default=10_000)
    a = ap.parse_args()
    if a.freeze:
        return freeze(a)
    if not a.out or not (len(a.run) == len(a.weights) == len(a.name)):
        raise SystemExit("--out and paired --run/--weights/--name required")
    score(a)


if __name__ == "__main__":
    main()
