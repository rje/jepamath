"""Score history-reversal items (scripts/reversal_items.py; design: docs/2026-11-26-history-reversal-design.md).

Per scene and checkpoint:
  - P_h: one prediction per history, of the post-reveal box tokens, from that history's own context (the frames
    before the screen starts lowering).
  - T_e: the target encoder on the post-reveal box tokens of ending e ONLY (keep indices). So a target sees no
    prefix, no hopper and no claw route.
  - S[h, e] = mean L1(P_h, T_e).
Primary endpoint: 4-way identification (argmin_e S[h, e] == h; ties count as wrong). Also reported:
  - the adjacent reversal rate;
  - the magnitude cells (H+ vs H++, H- vs H--);
  - margin by signed count difference.
The same endpoints are computed for cheap strategies from item metadata (oracles see one piece of the history and
break ties uniformly, so their numbers are expectations) and for a pixel strategy. Scene-cluster bootstrap,
stratified by n0.

    python scripts/reversal_score.py --items DIR --out OUT.json [--run RUN --weights W --name NAME]...
    python scripts/reversal_score.py --items DIR --out OUT.json --baselines-only
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from locate_surprise import MEAN, STD  # noqa: E402

TUB = 4


def load_items(d: str, expect_per_n0: int | None = None, min_seed: int = 0) -> list[dict]:
    items = [json.loads(l) for l in open(os.path.join(d, "items.jsonl"))]
    if not items:
        raise ValueError(f"no items in {d}")
    seeds = [it["seed"] for it in items]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{d}: duplicate scene seeds")
    if expect_per_n0 is not None:
        per = {n: sum(it["n0"] == n for it in items) for n in (1, 2)}
        if per != {1: expect_per_n0, 2: expect_per_n0} or min(seeds) < min_seed:
            raise ValueError(f"{d}: {per} scenes per n0 (want {expect_per_n0}), min seed {min(seeds)} (want >= {min_seed})")
    for it in items:
        z = np.load(os.path.join(d, f"scene_{it['seed']:09d}.npz"))
        it["frames"], it["box"] = z["frames"], z["box_tokens"]
        it["hopper_px"], it["hopper_ring"] = hopper_masks(it)
        if it["frames"].shape[0] != 4:
            raise ValueError(f"scene {it['seed']}: {it['frames'].shape[0]} histories")
    return items


def hopper_masks(it, res: int = 128, patch: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Pixel masks [res, res]: the projected hopper volume (outer walls up to rim + pile height) dilated by one
    patch, and a one-patch ring around that."""
    from matplotlib.path import Path
    from scipy.ndimage import binary_dilation
    from scipy.spatial import ConvexHull
    from jepagen.sampler import _project
    from jepagen.spec import Episode
    ep = Episode.model_validate(it["episodes"][it["histories"][0]])
    h, cam, c = ep.scene.hopper, ep.scene.camera, ep.clip
    hx, hy = h.inner_size[0] / 2 + h.wall_thickness, h.inner_size[1] / 2 + h.wall_thickness
    pts = np.array([(h.center[0] + sx * hx, h.center[1] + sy * hy, z) for sx in (-1, 1) for sy in (-1, 1)
                    for z in (0.0, h.wall_height + 0.14)])
    ndc = _project(cam, c.width / c.height, pts)
    px = np.stack([(ndc[:, 0] + 1) / 2 * res, (1 - ndc[:, 1]) / 2 * res], 1)
    yy, xx = np.mgrid[0:res, 0:res]
    inside = Path(px[ConvexHull(px).vertices]).contains_points(np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], 1))
    m = binary_dilation(inside.reshape(res, res), iterations=patch)
    ring = binary_dilation(m, iterations=patch) & ~m
    return m, ring


def finals(it) -> np.ndarray:
    return np.array([it["meta"][h]["final"] for h in it["histories"]])


# ---- endpoints on one score matrix S [4, 4] (rows: history, cols: ending; histories ordered by final) ----

def ident(S: np.ndarray, ties: str = "wrong") -> float:
    """Identification per row: the diagonal is the unique minimum (ties="wrong", the registered rule for models),
    or 1/|argmin set| if the diagonal is in it (ties="uniform", for the metadata oracles' expectations)."""
    if not np.isfinite(S).all():
        raise ValueError("non-finite score matrix")
    out = []
    for h in range(4):
        m = S[h] == S[h].min()
        out.append(float(m[h] and m.sum() == 1) if ties == "wrong" else m[h] / m.sum())
    return float(np.mean(out))


def reversal(S: np.ndarray, fin: np.ndarray) -> float:
    """Adjacent pairs (|final difference| = 1): both S[h,e] > S[h,h] and S[e,h] > S[e,e] (strict)."""
    r = [float(S[h, e] > S[h, h] and S[e, h] > S[e, e])
         for h, e in itertools.combinations(range(4), 2) if abs(fin[h] - fin[e]) == 1]
    return float(np.mean(r))


def magnitude(S: np.ndarray, names: list[str]) -> float | None:
    """Pairwise wins on the same-direction pairs (H+ vs H++, H- vs H--), both directions; ties 0.5."""
    pairs = [("H+", "H++"), ("H-", "H--")]
    r = []
    for a, b in pairs:
        if a in names and b in names:
            i, j = names.index(a), names.index(b)
            for h, e in ((i, j), (j, i)):
                r.append(1.0 if S[h, e] > S[h, h] else (0.5 if S[h, e] == S[h, h] else 0.0))
    return float(np.mean(r)) if r else None


def margins(S: np.ndarray, fin: np.ndarray) -> dict:
    """S[h, e] - S[h, h], keyed by final_e - final_h."""
    out: dict[int, list] = {}
    for h in range(4):
        for e in range(4):
            if e != h:
                out.setdefault(int(fin[e] - fin[h]), []).append(float(S[h, e] - S[h, h]))
    return out


def endpoints(mats: list[np.ndarray], items: list[dict], ties: str = "wrong") -> dict:
    per = []
    for S, it in zip(mats, items):
        fin = finals(it)
        per.append({"seed": it["seed"], "n0": it["n0"], "ident": ident(S, ties), "reversal": reversal(S, fin),
                    "magnitude": magnitude(S, it["histories"]), "margins": margins(S, fin)})
    return per


def summarize(per: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    strata = {n: [p for p in per if p["n0"] == n] for n in (1, 2)}
    if any(not v for v in strata.values()):
        raise ValueError(f"empty n0 stratum: { {k: len(v) for k, v in strata.items()} }")

    def stat(key, sample):
        # equal n0 weights
        vals = []
        for n, ps in sample.items():
            xs = [p[key] for p in ps if p[key] is not None]
            vals.append(np.mean(xs))
        return float(np.mean(vals))

    out = {}
    for key in ("ident", "reversal", "magnitude"):
        est = stat(key, strata)
        boots = [stat(key, {n: [ps[i] for i in rng.integers(0, len(ps), len(ps))] for n, ps in strata.items()})
                 for _ in range(n_boot)]
        out[key] = {"est": est, "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    md: dict[int, list] = {}
    for p in per:
        for k, v in p["margins"].items():
            md.setdefault(int(k), []).extend(v)
    out["margin_by_delta"] = {k: float(np.mean(v)) for k, v in sorted(md.items())}
    out["n_scenes"] = {n: len(v) for n, v in strata.items()}
    return out


def paired_diff(a: list[dict], b: list[dict], key: str, n_boot: int = 2000, seed: int = 1) -> dict:
    """Scene-paired difference a - b of an endpoint, equal n0 weights, stratified scene bootstrap."""
    if [p["seed"] for p in a] != [p["seed"] for p in b]:
        raise ValueError("paired_diff: scenes not aligned")
    d = {n: np.array([pa[key] - pb[key] for pa, pb in zip(a, b) if pa["n0"] == n and pa[key] is not None])
         for n in (1, 2)}
    rng = np.random.default_rng(seed)
    est = float(np.mean([v.mean() for v in d.values()]))
    boots = [np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in d.values()]) for _ in range(n_boot)]
    return {"est": est, "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


# ---- cheap strategies ----

def oracle_matrix(it, key) -> np.ndarray:
    """0 where ending e agrees with history h on the information `key` reads, else 1. key None: the
    last-visible-count strategy, |final_e - n0| for every row."""
    if key is None:
        fin = finals(it)
        return np.tile(np.abs(fin - it["n0"]).astype(float), (4, 1))
    info = [key(it["meta"][h], it["n0"]) for h in it["histories"]]
    return np.array([[0.0 if info[h] == info[e] else 1.0 for e in range(4)] for h in range(4)])


ORACLES = {
    "history_blind": lambda m, n0: 0,
    "last_visible_count": None,                # rows of |final_e - n0|: always picks H0 (oracle_matrix)
    "first_transfer": lambda m, n0: tuple(m["route"][0]),
    "last_transfer": lambda m, n0: tuple(m["route"][1]),
    "sign": lambda m, n0: int(np.sign(m["final"] - n0)),
    "count": lambda m, n0: m["final"],
}


def pixel_matrix(it) -> np.ndarray:
    """Pixel persistence on the box tokens: L1 between history h's last context frame and ending e's mean
    post-reveal frame. The screen hides the box in context, so this mainly checks that endings don't leak
    through the occluder."""
    fr = it["frames"].astype(np.float32) / 255
    m = np.kron(it["box"], np.ones((8, 8), bool))
    ctx, first = it["ctx"], it["first_post_slot"]
    last_ctx = fr[:, ctx - 1][:, m]                                # [4, P, 3]
    post = fr[:, ctx + first * TUB:][:, :, m]                       # [4, F, P, 3]
    return np.abs(last_ctx[:, None, None] - post[None]).mean((2, 3, 4))   # L1 per frame, then averaged


# ---- model ----

def mask_hopper(it) -> np.ndarray:
    """Frames with the hopper region painted out in the context frames (H-C2 arm): the projected hopper
    volume, dilated by one patch, filled with the median colour of a one-patch ring around it in frame 0.
    Without it, box = total - hopper is readable from the last context frame with no transfer seen."""
    fr = it["frames"].copy()
    m, ring = it["hopper_px"], it["hopper_ring"]
    fill = np.median(fr[0, 0][ring], 0).astype(fr.dtype)
    fr[:, : it["ctx"]][:, :, m] = fill
    return fr


@torch.no_grad()
def model_matrix(it, enc, tgt, pred, full_target: bool = False, hopper_masked: bool = False) -> np.ndarray:
    frames = mask_hopper(it) if hopper_masked else it["frames"]
    x = torch.from_numpy(frames).permute(0, 4, 1, 2, 3).float() / 255      # [4, 3, T, H, W]
    x = (x - MEAN) / STD
    G = enc.grid * enc.grid
    ctx, first = it["ctx"], it["first_post_slot"]
    if ctx % TUB:
        raise ValueError("ctx not a tubelet multiple")
    slots = x.shape[2] // TUB
    s0 = ctx // TUB + first
    box = torch.from_numpy(np.flatnonzero(it["box"].ravel()))
    keep = torch.cat([s * G + box for s in range(s0, slots)])                 # post-reveal box tokens
    n_ctx = ctx // TUB * G
    z = enc(x[:, :, :ctx])
    ctx_idx = torch.arange(n_ctx).unsqueeze(0).expand(4, -1)
    N = slots * G
    fut = torch.arange(n_ctx, N).unsqueeze(0).expand(4, -1)
    P = pred(z, ctx_idx, fut).float()[:, keep - n_ctx]                        # every future token queried, as in prefix training
    if full_target:
        h = tgt(x).float()
        T = F.layer_norm(h, (h.size(-1),))[:, keep]
    else:
        h = tgt(x, keep.unsqueeze(0).expand(4, -1)).float()
        T = F.layer_norm(h, (h.size(-1),))
    return (P[:, None] - T[None]).abs().mean((2, 3)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--weights", action="append", default=[])
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--expect-per-n0", type=int, default=None, help="confirmatory set: exact scenes per n0")
    ap.add_argument("--min-seed", type=int, default=0)
    a = ap.parse_args()
    if not (len(a.run) == len(a.weights) == len(a.name)):
        raise SystemExit("--run, --weights and --name must pair up")
    if not a.run and not a.baselines_only:
        raise SystemExit("give --run/--weights/--name or --baselines-only")
    items = load_items(a.items, a.expect_per_n0, a.min_seed)
    res = {"items": a.items, "n_items": len(items), "baselines": {}, "models": {}}
    for k, key in ORACLES.items():
        res["baselines"][k] = summarize(endpoints([oracle_matrix(it, key) for it in items], items, ties="uniform"))
    res["baselines"]["pixel"] = summarize(endpoints([pixel_matrix(it) for it in items], items))
    oracle_per = {"last_transfer": endpoints([oracle_matrix(it, ORACLES["last_transfer"]) for it in items], items, ties="uniform")}
    for k, v in res["baselines"].items():
        print(f"[baseline] {k:20s} ident {v['ident']['est']:.3f} reversal {v['reversal']['est']:.3f} "
              f"magnitude {v['magnitude']['est']:.3f}", flush=True)
    from jepatrain.reeval import load_run
    torch.set_num_threads(16)
    for run, w, name in zip(a.run, a.weights, a.name):
        enc, tgt, pred, step = load_run(run, w, device="cpu")
        enc.eval(); tgt.eval(); pred.eval()
        mats = [model_matrix(it, enc, tgt, pred) for it in items]
        per = endpoints(mats, items)
        r = summarize(per)
        r["magnitude_vs_last_transfer"] = paired_diff(per, oracle_per["last_transfer"], "magnitude")
        r["ident_vs_last_transfer"] = paired_diff(per, oracle_per["last_transfer"], "ident")
        r["hopper_masked"] = summarize(endpoints([model_matrix(it, enc, tgt, pred, hopper_masked=True) for it in items], items))
        r["full_target_exploratory"] = summarize(endpoints([model_matrix(it, enc, tgt, pred, True) for it in items], items))
        r["run"], r["weights"], r["step"] = run, w, step
        res["models"][name] = r
        print(f"[{name}] ident {r['ident']['est']:.3f} {r['ident']['ci']}  reversal {r['reversal']['est']:.3f} "
              f"{r['reversal']['ci']}  magnitude {r['magnitude']['est']:.3f}  margins {r['margin_by_delta']}  "
              f"| hopper-masked ident {r['hopper_masked']['ident']['est']:.3f} {r['hopper_masked']['ident']['ci']} "
              f"| full-target ident {r['full_target_exploratory']['ident']['est']:.3f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
