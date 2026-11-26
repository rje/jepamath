"""P1: donor-free count probe on the predictor's hidden-ending tokens (research log 27 Nov, pre-registered).

Features per source history: hopper-masked context -> predictor queried with all future tokens -> the predicted
tokens at the source's post-reveal box tokens, mean-pooled (dim). Visibility control: the target encoder's
layer-normed tokens of the history's ACTUAL revealed ending at the same tokens, pooled.
Fit: ridge regression of final count on C8 sources (counts from the simulator records), alpha by 5-fold CV grouped
by block. Test: C9 edges (L -> H correct if c_hat(H) > c_hat(L); half credit for ties), donor-free; shape-stratified
block bootstrap. C9 never enters fitting or tuning.

    python scripts/p1_count_probe.py --fit-manifest C8.json --test-manifest C9.json --out OUT.json --run R --weights W --name N [...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from c6_score import actual_finals, load_model, mask_hopper  # noqa: E402
from c9_items import FAMILIES  # noqa: E402
from c9_score import EDGES, TIE, bootstrap, final_objects  # noqa: E402
from locate_surprise import MEAN, STD  # noqa: E402

TUB = 4


@torch.no_grad()
def feats(frames: np.ndarray, masked: np.ndarray, ctx: int, first: int, box: np.ndarray, enc, tgt, pred):
    """(predicted pooled [n, dim], revealed pooled [n, dim], predicted per-box-token [n, n_box, dim] averaged over the
    post-reveal slots, the same for revealed) for n clips of one scene."""
    def prep(fr):
        x = torch.from_numpy(fr).permute(0, 4, 1, 2, 3).float() / 255
        return (x - MEAN) / STD
    xm, xf = prep(masked), prep(frames)
    G = enc.grid * enc.grid
    slots = xm.shape[2] // TUB
    s0 = ctx // TUB + first
    tok = torch.from_numpy(np.flatnonzero(box.ravel()))
    keep = torch.cat([s * G + tok for s in range(s0, slots)])
    n_ctx, N = ctx // TUB * G, slots * G
    n = len(frames)
    z = enc(xm[:, :, :ctx])
    P = pred(z, torch.arange(n_ctx).unsqueeze(0).expand(n, -1), torch.arange(n_ctx, N).unsqueeze(0).expand(n, -1)).float()
    P = P[:, keep - n_ctx]
    h = tgt(xf, keep.unsqueeze(0).expand(n, -1)).float()
    T = F.layer_norm(h, (h.size(-1),))
    nb = len(tok)
    Pt = P.reshape(n, -1, nb, P.size(-1)).mean(1)
    Tt = T.reshape(n, -1, nb, T.size(-1)).mean(1)
    return P.mean(1).numpy(), T.mean(1).numpy(), Pt, Tt


class MaskedAttn(nn.Module):
    """4 learned queries over a padded token set, then a 3-way classifier (counts 1-3)."""

    def __init__(self, d, q=4, k=3):
        super().__init__()
        self.q = nn.Parameter(torch.randn(q, d) / math.sqrt(d))
        self.kp, self.vp = nn.Linear(d, d), nn.Linear(d, 64)
        self.out = nn.Linear(q * 64, k)

    def forward(self, x, mask):                     # x [B, G, D], mask [B, G] True = real token
        logit = self.kp(x) @ self.q.T / math.sqrt(x.shape[-1])
        logit = logit.masked_fill(~mask[..., None], -1e9)
        a = torch.softmax(logit, dim=1)
        v = torch.einsum("bgq,bgd->bqd", a, self.vp(x))
        return self.out(v.flatten(1))


def pad(tokens: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    G = max(t.shape[0] for t in tokens)
    X = torch.zeros(len(tokens), G, tokens[0].shape[-1])
    M = torch.zeros(len(tokens), G, dtype=torch.bool)
    for i, t in enumerate(tokens):
        X[i, :t.shape[0]] = t
        M[i, :t.shape[0]] = True
    return X, M


def fit_attn(tokens: list[torch.Tensor], y: np.ndarray, epochs=300, lr=3e-3, wd=1e-3, seed=0) -> MaskedAttn:
    torch.manual_seed(seed)
    X, M = pad(tokens)
    yy = torch.from_numpy(y.astype(np.int64) - 1)
    m = MaskedAttn(X.shape[-1])
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        for i in torch.randperm(len(X)).split(128):
            opt.zero_grad()
            F.cross_entropy(m(X[i], M[i]), yy[i]).backward()
            opt.step()
    return m


@torch.no_grad()
def attn_count(m: MaskedAttn, tokens: list[torch.Tensor]) -> np.ndarray:
    X, M = pad(tokens)
    p = torch.softmax(m(X, M), 1)
    return (p * torch.tensor([1.0, 2.0, 3.0])).sum(1).numpy()


def c8_items(man):
    srec = {json.loads(l)["seed"]: json.loads(l) for l in open(os.path.join(man["sources_dir"], "items.jsonl"))}
    for p in man["pairs"]:
        s = srec[p["source"]]
        z = np.load(os.path.join(man["sources_dir"], f"source_{p['source']:09d}.npz"))
        yield p["source"], s, z, actual_finals(s).astype(float)


def c9_items(man):
    srec = {json.loads(l)["seed"]: json.loads(l) for l in open(os.path.join(man["sources_dir"], "items.jsonl"))}
    for b in man["blocks"]:
        s = srec[b["source"]]
        z = np.load(os.path.join(man["sources_dir"], f"source_{b['source']:09d}.npz"))
        fin = np.array([len(final_objects(s, n)) for n in s["names"]], float)
        yield b, s, z, fin


def ridge_cv(X, y, groups, alphas=(0.1, 1, 10, 100, 1000)):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    best = None
    for al in alphas:
        err = []
        for tr, te in GroupKFold(5).split(X, y, groups):
            m = make_pipeline(StandardScaler(), Ridge(alpha=al)).fit(X[tr], y[tr])
            err.append(np.mean((m.predict(X[te]) - y[te]) ** 2))
        if best is None or np.mean(err) < best[0]:
            best = (float(np.mean(err)), al)
    m = make_pipeline(StandardScaler(), Ridge(alpha=best[1])).fit(X, y)
    return m, {"alpha": best[1], "cv_mse": best[0], "var_y": float(np.var(y))}


def edges(chat: dict) -> dict:
    out = {}
    for e, lo in EDGES.items():
        for fam in FAMILIES:
            hi_, lo_ = chat[f"{fam}:H"], chat[f"{fam}:{lo}"]
            out[f"{e}:{fam}"] = 0.5 if abs(hi_ - lo_) <= TIE else float(hi_ > lo_)
        out[f"pair_area:{e}"] = (out[f"{e}:orig"] + out[f"{e}:area"]) / 2
        out[f"pair_volume:{e}"] = (out[f"{e}:orig"] + out[f"{e}:volume"]) / 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-manifest", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--weights", action="append", default=[])
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--boot", type=int, default=10_000)
    a = ap.parse_args()
    fm, tm = json.load(open(a.fit_manifest)), json.load(open(a.test_manifest))
    torch.set_num_threads(16)
    res, per, strata = {"fit": a.fit_manifest, "test": a.test_manifest, "models": {}}, {}, None
    for run, w, name in zip(a.run, a.weights, a.name):
        enc, tgt, pred, step = load_model(run, w)
        enc.eval(); tgt.eval(); pred.eval()
        Xp, Xt, y, g, Tp, Tt = [], [], [], [], [], []
        for seed, s, z, fin in c8_items(fm):
            m = mask_hopper(z["frames"], z["hopper_px"], z["hopper_ring"], s["ctx"])
            P, T, pt, tt = feats(z["frames"], m, s["ctx"], s["first_post_slot"], z["box_tokens"].astype(bool), enc, tgt, pred)
            Xp.append(P); Xt.append(T); y.append(fin); g += [seed] * len(fin)
            Tp += list(pt); Tt += list(tt)
        Xp, Xt, y, g = np.concatenate(Xp), np.concatenate(Xt), np.concatenate(y), np.array(g)
        mp, ip = ridge_cv(Xp, y, g)
        mt, it = ridge_cv(Xt, y, g)
        ap_, at_ = fit_attn(Tp, y), fit_attn(Tt, y)
        rows = {k: [] for k in ("pred", "revealed", "attn_pred", "attn_revealed")}
        sh = []
        for b, s, z, fin in c9_items(tm):
            m = mask_hopper(z["frames"], z["hopper_px"], z["hopper_ring"], s["ctx"])
            P, T, pt, tt = feats(z["frames"], m, s["ctx"], s["first_post_slot"], z["box_tokens"].astype(bool), enc, tgt, pred)
            for k, c in (("pred", mp.predict(P)), ("revealed", mt.predict(T)),
                         ("attn_pred", attn_count(ap_, list(pt))), ("attn_revealed", attn_count(at_, list(tt)))):
                rows[k].append(edges(dict(zip(s["names"], c))))
            sh.append(b["shape"])
        strata = np.array(sh)
        for k, r in rows.items():
            per[f"{name}:{k}"] = r
        rows_p, rows_t = rows["pred"], rows["revealed"]
        print(f"[{name}] attn " + "  ".join(f"{k} p {np.mean([r[k] for r in rows['attn_pred']]):.3f} "
                                            f"r {np.mean([r[k] for r in rows['attn_revealed']]):.3f}"
                                            for k in ("conflict:orig", "pair_area:conflict", "pair_volume:conflict")), flush=True)
        res["models"][name] = {"run": run, "weights": w, "step": step, "fit_pred": ip, "fit_revealed": it}
        print(f"[{name}] fit pred cv_mse {ip['cv_mse']:.3f} (var {ip['var_y']:.3f}), revealed {it['cv_mse']:.3f} | "
              + "  ".join(f"{k} p {np.mean([r[k] for r in rows_p]):.3f} r {np.mean([r[k] for r in rows_t]):.3f}"
                          for k in ("conflict:orig", "pair_area:conflict", "pair_volume:conflict", "Ladd:orig")), flush=True)
    names = list(res["models"])
    pairs = [(f"{m}:{k}", f"{u}:{k}") for k in ("pred", "attn_pred") for m in names for u in names if m != u]
    res["bootstrap"] = bootstrap(per, strata, a.boot, pairs)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
