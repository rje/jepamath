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


def load_items(d: str) -> list[dict]:
    items = [json.loads(l) for l in open(os.path.join(d, "items.jsonl"))]
    if not items:
        raise ValueError(f"no items in {d}")
    for it in items:
        z = np.load(os.path.join(d, f"scene_{it['seed']:09d}.npz"))
        it["frames"], it["box"] = z["frames"], z["box_tokens"]
        if it["frames"].shape[0] != 4:
            raise ValueError(f"scene {it['seed']}: {it['frames'].shape[0]} histories")
    return items


def finals(it) -> np.ndarray:
    return np.array([it["meta"][h]["final"] for h in it["histories"]])


# ---- endpoints on one score matrix S [4, 4] (rows: history, cols: ending; histories ordered by final) ----

def ident(S: np.ndarray) -> float:
    """Expected identification: for each row, 1/|argmin set| if the diagonal is in it (uniform tie-break).
    A model's real-valued scores have no ties, so this is the plain hit rate."""
    out = []
    for h in range(4):
        m = S[h] == S[h].min()
        out.append(m[h] / m.sum())
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


def endpoints(mats: list[np.ndarray], items: list[dict]) -> dict:
    per = []
    for S, it in zip(mats, items):
        fin = finals(it)
        per.append({"n0": it["n0"], "ident": ident(S), "reversal": reversal(S, fin),
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


# ---- cheap strategies ----

def oracle_matrix(it, key) -> np.ndarray:
    """0 where ending e agrees with history h on the information `key` reads, else 1."""
    info = [key(it["meta"][h], it["n0"]) for h in it["histories"]]
    return np.array([[0.0 if info[h] == info[e] else 1.0 for e in range(4)] for h in range(4)])


ORACLES = {
    "history_blind": lambda m, n0: 0,
    "last_visible_count": lambda m, n0: 0,    # predicts n0 for every history: history-blind
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
    post = fr[:, ctx + first * TUB:].mean(1)[:, m]                  # [4, P, 3]
    return np.abs(last_ctx[:, None] - post[None]).mean((2, 3))


# ---- model ----

@torch.no_grad()
def model_matrix(it, enc, tgt, pred, full_target: bool = False) -> np.ndarray:
    x = torch.from_numpy(it["frames"]).permute(0, 4, 1, 2, 3).float() / 255      # [4, 3, T, H, W]
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
    P = pred(z, ctx_idx, keep.unsqueeze(0).expand(4, -1)).float()             # [4, K, D]
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
    a = ap.parse_args()
    if not (len(a.run) == len(a.weights) == len(a.name)):
        raise SystemExit("--run, --weights and --name must pair up")
    if not a.run and not a.baselines_only:
        raise SystemExit("give --run/--weights/--name or --baselines-only")
    items = load_items(a.items)
    res = {"items": a.items, "n_items": len(items), "baselines": {}, "models": {}}
    for k, key in ORACLES.items():
        res["baselines"][k] = summarize(endpoints([oracle_matrix(it, key) for it in items], items))
    res["baselines"]["pixel"] = summarize(endpoints([pixel_matrix(it) for it in items], items))
    for k, v in res["baselines"].items():
        print(f"[baseline] {k:20s} ident {v['ident']['est']:.3f} reversal {v['reversal']['est']:.3f} "
              f"magnitude {v['magnitude']['est']:.3f}", flush=True)
    from jepatrain.reeval import load_run
    torch.set_num_threads(16)
    for run, w, name in zip(a.run, a.weights, a.name):
        enc, tgt, pred, step = load_run(run, w, device="cpu")
        enc.eval(); tgt.eval(); pred.eval()
        mats = [model_matrix(it, enc, tgt, pred) for it in items]
        r = summarize(endpoints(mats, items))
        r["full_target_exploratory"] = summarize(endpoints([model_matrix(it, enc, tgt, pred, True) for it in items], items))
        r["run"], r["weights"], r["step"] = run, w, step
        res["models"][name] = r
        print(f"[{name}] ident {r['ident']['est']:.3f} {r['ident']['ci']}  reversal {r['reversal']['est']:.3f} "
              f"{r['reversal']['ci']}  magnitude {r['magnitude']['est']:.3f}  margins {r['margin_by_delta']}  "
              f"| full-target ident {r['full_target_exploratory']['ident']['est']:.3f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
