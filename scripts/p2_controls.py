"""P2: controls for P1 (research log 27 Nov, pre-registered).

Ridge probes of the final count, fit on C8 sources and scored on C9 edges exactly as P1, on four feature sets:
  pixel    : hopper-masked context frames average-pooled to 16x16 RGB per frame (model-free; a leak check)
  enc_last : context-encoder tokens of the last context slot at the source's box tokens, mean-pooled
  enc_all  : context-encoder tokens over all context slots, mean-pooled
  pred     : P1's feature (predicted post-reveal box tokens, mean-pooled), recomputed
Steering (objlabel models): shift every predicted token by delta * u, u = w / |w|^2 with w the pred ridge direction in raw
feature space (so the probe's count moves by exactly delta), and recompute C9's L1 preference
p = d(2-object ending) - d(3-object ending) over the donor's common tokens. S = share of (history, layout) cells with
p(+1) > p(-1); S_rand = the same for random directions of equal norm.

    CUDA_VISIBLE_DEVICES= python scripts/p2_controls.py --fit-manifest C8.json --test-manifest C9.json --out OUT.json \
        --run R --weights W --name N [--steer N ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from c6_score import load_model, mask_hopper  # noqa: E402
from c9_score import bootstrap  # noqa: E402
from locate_surprise import MEAN, STD  # noqa: E402
from p1_count_probe import c8_items, c9_items, edges, ridge_cv  # noqa: E402

TUB = 4
DELTAS = (-2, -1, 0, 1, 2)
N_RAND = 5


def prep(fr: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(fr).permute(0, 4, 1, 2, 3).float() / 255
    return (x - MEAN) / STD


def pixel_feat(masked: np.ndarray, ctx: int) -> np.ndarray:
    """16x16 RGB per frame, then 12 time bins over the context (ctx varies across items: C6 retime)."""
    x = torch.from_numpy(masked[:, :ctx]).float() / 255                       # [n, ctx, H, W, 3]
    n, t, H, W, c = x.shape
    x = F.adaptive_avg_pool3d(x.permute(0, 4, 1, 2, 3), (12, 16, 16))          # [n, 3, 12, 16, 16]
    return x.reshape(n, -1).numpy()


@torch.no_grad()
def model_feats(masked: np.ndarray, ctx: int, first: int, box: np.ndarray, enc, pred):
    """enc_last [n, D], enc_all [n, D], pred pooled [n, D], and the predictor's full future output P [n, N-n_ctx, D]."""
    xm = prep(masked)
    G = enc.grid * enc.grid
    slots = xm.shape[2] // TUB
    n_ctx, N = ctx // TUB * G, slots * G
    n = len(masked)
    tok = torch.from_numpy(np.flatnonzero(box.ravel()))
    z = enc(xm[:, :, :ctx])                                                     # [n, n_ctx, D]
    last = (ctx // TUB - 1) * G + tok
    P = pred(z, torch.arange(n_ctx).unsqueeze(0).expand(n, -1), torch.arange(n_ctx, N).unsqueeze(0).expand(n, -1)).float()
    s0 = ctx // TUB + first
    keep = torch.cat([s * G + tok for s in range(s0, slots)])
    return (z[:, last].float().mean(1).numpy(), z.float().mean(1).numpy(), P[:, keep - n_ctx].mean(1).numpy(), P)


@torch.no_grad()
def donor_targets(xd: np.ndarray, ctx: int, first: int, common: np.ndarray, enc, tgt):
    """Layer-normed target tokens of the donor's two endings at the common post-reveal tokens, and the indices of
    those tokens inside the predictor's future output."""
    b = prep(xd)
    G = enc.grid * enc.grid
    slots = b.shape[2] // TUB
    s0 = ctx // TUB + first
    tok = torch.from_numpy(np.asarray(common))
    keep = torch.cat([s * G + tok for s in range(s0, slots)])
    h = tgt(b, keep.unsqueeze(0).expand(len(xd), -1)).float()
    return F.layer_norm(h, (h.size(-1),)), keep - ctx // TUB * G


def pref(P: torch.Tensor, T: torch.Tensor) -> np.ndarray:
    """p = d(col 0 = 2-object ending) - d(col 1 = 3-object ending), as C9's model_matrix (mean L1)."""
    M = (P[:, None] - T[None]).abs().mean((2, 3))
    return (M[:, 0] - M[:, 1]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-manifest", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--weights", action="append", default=[])
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--steer", action="append", default=[], help="model names to run the steering test on")
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--limit", type=int, default=0, help="smoke test only: first N fit and test items")
    a = ap.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("run with CUDA_VISIBLE_DEVICES=")
    if not (len(a.run) == len(a.weights) == len(a.name)):
        raise SystemExit("--run/--weights/--name must pair up")
    if set(a.steer) - set(a.name):
        raise SystemExit(f"--steer names not among --name: {set(a.steer) - set(a.name)}")
    fm, tm = json.load(open(a.fit_manifest)), json.load(open(a.test_manifest))
    torch.set_num_threads(16)
    lim = (lambda it: (x for i, x in zip(range(a.limit), it))) if a.limit else (lambda it: it)
    res = {"fit": a.fit_manifest, "test": a.test_manifest, "deltas": DELTAS, "models": {}}
    per, strata = {}, None

    # ---- model-free pixel probe (computed once) ----
    X, y, g = [], [], []
    for seed, s, z, fin in lim(c8_items(fm)):
        X.append(pixel_feat(mask_hopper(z["frames"], z["hopper_px"], z["hopper_ring"], s["ctx"]), s["ctx"]))
        y.append(fin); g += [seed] * len(fin)
    mpx, ipx = ridge_cv(np.concatenate(X), np.concatenate(y), np.array(g))
    res["fit_pixel"] = ipx
    rows, sh = [], []
    for b, s, z, fin in lim(c9_items(tm)):
        c = mpx.predict(pixel_feat(mask_hopper(z["frames"], z["hopper_px"], z["hopper_ring"], s["ctx"]), s["ctx"]))
        rows.append(edges(dict(zip(s["names"], c))))
        sh.append(b["shape"])
    per["pixel"], strata = rows, np.array(sh)
    print(f"[pixel] cv_mse {ipx['cv_mse']:.3f} (var {ipx['var_y']:.3f}) | "
          + "  ".join(f"{k} {np.mean([r[k] for r in rows]):.3f}" for k in ("conflict:orig", "Ladd:orig", "LN:orig")), flush=True)

    for run, w, name in zip(a.run, a.weights, a.name):
        enc, tgt, pred, step = load_model(run, w)
        enc.eval(); tgt.eval(); pred.eval()
        F_ = {k: [] for k in ("enc_last", "enc_all", "pred")}
        y, g = [], []
        for seed, s, z, fin in lim(c8_items(fm)):
            m = mask_hopper(z["frames"], z["hopper_px"], z["hopper_ring"], s["ctx"])
            el, ea, pp, _ = model_feats(m, s["ctx"], s["first_post_slot"], z["box_tokens"].astype(bool), enc, pred)
            F_["enc_last"].append(el); F_["enc_all"].append(ea); F_["pred"].append(pp)
            y.append(fin); g += [seed] * len(fin)
        y, g = np.concatenate(y), np.array(g)
        probes, info = {}, {}
        for k, v in F_.items():
            probes[k], info[k] = ridge_cv(np.concatenate(v), y, g)
        # steering direction: the pred probe's weight in raw feature space; u moves the probe's count by exactly 1
        sc, rd = probes["pred"].named_steps["standardscaler"], probes["pred"].named_steps["ridge"]
        wr = torch.from_numpy(rd.coef_ / sc.scale_).float()
        u = wr / (wr @ wr)
        rng = np.random.default_rng(0)
        rand = []
        for _ in range(N_RAND):
            r = torch.from_numpy(rng.standard_normal(len(u))).float()
            rand.append(r / r.norm() * u.norm())
        steer = name in a.steer
        rows = {k: [] for k in F_}
        srows, dp = [], []
        for b, s, z, fin in lim(c9_items(tm)):
            m = mask_hopper(z["frames"], z["hopper_px"], z["hopper_ring"], s["ctx"])
            el, ea, pp, P = model_feats(m, s["ctx"], s["first_post_slot"], z["box_tokens"].astype(bool), enc, pred)
            for k, feat in (("enc_last", el), ("enc_all", ea), ("pred", pp)):
                rows[k].append(edges(dict(zip(s["names"], probes[k].predict(feat)))))
            if steer:
                cell, cell_r, shift = [], [], []
                for lay in ("rev", "fwd"):
                    zd = np.load(os.path.join(tm[f"{lay}_dir"], f"donor_{b[lay]['donor']:09d}.npz"))
                    T, idx = donor_targets(zd["frames"], s["ctx"], s["first_post_slot"], np.array(b[lay]["common"]), enc, tgt)
                    Pc = P[:, idx]
                    pd = {d: pref(Pc + d * u, T) for d in DELTAS}
                    cell.append((pd[1] > pd[-1]).astype(float))
                    shift.append((pd[1] - pd[-1]) / 2)
                    for r in rand:
                        cell_r.append((pref(Pc + r, T) > pref(Pc - r, T)).astype(float))
                    dp.append({d: pd[d].tolist() for d in DELTAS})
                srows.append({"S": float(np.mean(cell)), "S_rand": float(np.mean(cell_r)),
                              "S_minus_rand": float(np.mean(cell) - np.mean(cell_r)),
                              "shift_per_unit": float(np.mean(shift)),
                              "p_sd": float(np.std(np.concatenate([np.array(d[0]) for d in dp[-2:]])))})
        for k, r in rows.items():
            per[f"{name}:{k}"] = r
        if steer:
            per[f"{name}:steer"] = srows
            res.setdefault("steer_p", {})[name] = dp
        res["models"][name] = {"run": run, "weights": w, "step": step, "fit": info, "u_norm": float(u.norm())}
        print(f"[{name}] " + "  ".join(f"{k} cv {info[k]['cv_mse']:.3f}: conflict {np.mean([x['conflict:orig'] for x in r]):.3f} "
                                       f"Ladd {np.mean([x['Ladd:orig'] for x in r]):.3f}" for k, r in rows.items()), flush=True)
        if steer:
            print(f"[{name}] steer " + "  ".join(f"{k} {np.mean([x[k] for x in srows]):.4f}" for k in srows[0]), flush=True)

    # bootstrap the probe rows and the steering rows separately (different endpoint keys)
    probe_per = {k: v for k, v in per.items() if not k.endswith(":steer")}
    names = list(res["models"])
    pairs = [(f"{m}:pred", f"{m}:{k}") for m in names for k in ("enc_last", "enc_all")] + \
            [(f"{m}:{k}", "pixel") for m in names for k in ("enc_last", "enc_all", "pred")]
    res["bootstrap"] = bootstrap(probe_per, strata, a.boot, pairs)
    steer_per = {k: v for k, v in per.items() if k.endswith(":steer")}
    if steer_per:
        res["bootstrap_steer"] = bootstrap(steer_per, strata, a.boot, [])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
