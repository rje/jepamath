"""Object sense in a frozen encoder, before any counting: are objects there as units, with kind, size, depth?

On fresh probe clips (whole clip, all slots, the online encoder), scored against the renderer's per-token object
labels (jepatrain.data.token_seg; claw and screen count as background) and its per-object state.
  grouping  : over token pairs in one slot, both on objects, AUC of cosine similarity for "same object" vs
              "different objects", matched for distance (AUC within Chebyshev-distance bins, then averaged by
              pair count). `grouping_hard`: only different-object pairs whose mean pixel colours are close
              (bottom quartile), so colour alone cannot separate them.
  identity  : each object's mean token at slot s vs slot s + gap; accuracy of matching every object to itself
              among the objects visible in both (chance = mean 1/K). `identity_moved`: only objects whose centre
              moved >= 2 tokens, so position alone cannot match them.
  figure    : logistic probe, object vs background token, balanced accuracy on held-out clips.
  kind      : logistic probe, shape (sphere / box / cylinder) from an object's mean token in the last slot.
  size      : ridge probe, log cube-root volume of the object from its mean token (R^2).
              kind, size and the 3D probes standardise features and pick their regularisation by 5-fold CV on the
              training clips only. (No extent probe: at 128 px / patch 8 objects are ~2 x 2 tokens, so nearly every
              object token is on the edge; checked 25 Nov, 2364 of 2367 tokens at distance 1.)
  depth3d / height3d : ridge probes on the object's mean token for its camera depth (along the view axis) and the
              height of its bottom above the table, each first residualised on the object's image centroid and token
              area (a linear fit on the training clips), so image position and apparent size cannot carry them.
Probes fit on the first 70% of clips and score the rest. `--boot B` adds 95% bootstrap CIs over clips (test clips for
probes): every metric gets [lo, hi]. Each is reported at the embedding (layer 0: appearance + position, the
no-learning baseline) and the chosen layers.

    CUDA_VISIBLE_DEVICES= python scripts/object_sense.py RUN --weights W --out OUT.json [--boot 1000]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import webdataset as wds
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from jepagen.sampler import _volume, _z_extent  # noqa: E402
from jepagen.shards import load_sample  # noqa: E402
from jepatrain.data import downsample, token_seg  # noqa: E402
from jepatrain.reeval import load_run  # noqa: E402
from locate_surprise import MEAN, STD  # noqa: E402

DIST_BINS = (1, 2, 3, 5, 9, 99)          # Chebyshev distance bins on the token grid: [1,2), [2,3), [3,5), ...
SHAPES = ("sphere", "box", "cylinder")


@torch.no_grad()
def layer_tokens(enc, x, layers):
    """x [1, 3, T, H, W] -> {layer: [N, D]} for the whole clip; layer 0 = embedding + position."""
    t = enc.embed(x).flatten(2).transpose(1, 2)
    t = t + enc.pos[: t.shape[1]]
    out = {0: t[0]} if 0 in layers else {}
    for li, b in enumerate(enc.blocks, 1):
        t = b(t)
        if li in layers:
            out[li] = (enc.norm(t) if li == len(enc.blocks) else t)[0]
    return out


class MatchedAUC:
    """AUC of sim for same vs different object, within distance bins, averaged by (weighted) pair count over bins
    that hold both kinds. Sorting is done once, so each bootstrap draw (new pair weights) costs O(pairs)."""

    def __init__(self, sim, same, dist):
        self.bins = []
        for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:]):
            k = (dist >= lo) & (dist < hi)
            ip, ineg = np.flatnonzero(k & same), np.flatnonzero(k & ~same)
            if len(ip) == 0 or len(ineg) == 0:
                continue
            order = np.argsort(sim[ineg], kind="stable"); ineg = ineg[order]; sn = sim[ineg]
            self.bins.append((ip, ineg, np.searchsorted(sn, sim[ip], "left"), np.searchsorted(sn, sim[ip], "right")))

    def __call__(self, w):
        tot, wt = 0.0, 0.0
        for ip, ineg, lo, hi in self.bins:
            wpos, wneg = w[ip], w[ineg]
            if wpos.sum() == 0 or wneg.sum() == 0:
                continue
            cw = np.r_[0.0, np.cumsum(wneg)]                      # ties count half
            a = (wpos * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))).sum() / (wpos.sum() * wneg.sum())
            nb = wpos.sum() + wneg.sum()
            tot += a * nb; wt += nb
        return tot / wt if wt else float("nan")


def auc_matched(sim, same, dist, w=None):
    """One-shot MatchedAUC (unit weights unless given)."""
    return MatchedAUC(sim, same, dist)(np.ones(len(sim)) if w is None else w)


def _r2(y, p, w):
    mu = (w * y).sum() / w.sum()
    return float(1 - (w * (y - p) ** 2).sum() / (w * (y - mu) ** 2).sum())


def _bal(y, p, w):
    return float(np.mean([(w[y == c] * (p[y == c] == c)).sum() / w[y == c].sum() for c in np.unique(y)]))


def collect(enc, res, data, n_clips, layers, gap=4, pairs_per_slot=300):
    """Per-clip raw records for every metric, for every layer."""
    G, S, P, tb = enc.grid, enc.slots, enc.patch, enc.tubelet
    rng = np.random.default_rng(0)
    yy, xx = np.divmod(np.arange(G * G), G)
    R = {"pairs": {L: [] for L in layers}, "pair_clip": [], "pair_meta": [],
         "ident": {L: [] for L in layers}, "ident_meta": [],
         "fig": {L: [] for L in layers}, "fig_y": [], "fig_clip": [],
         "obj": {L: [] for L in layers}, "obj_meta": []}          # per object, last slot: kind, size, 3D
    n = 0
    for s in wds.WebDataset(sorted(glob.glob(os.path.join(data, "train", "possible", "*.tar"))),
                            shardshuffle=False, empty_check=False):
        ep, frames, seg, state = load_sample(s)
        fr = downsample(frames, res)
        if fr.shape[0] != S * tb:
            raise ValueError(f"clip has {fr.shape[0]} frames, encoder wants {S * tb}")
        lab = token_seg(seg, res, P, tb).reshape(S, G * G).astype(int)
        col = fr.reshape(S, tb, G, P, G, P, 3).mean((1, 3, 5)).reshape(S, G * G, 3)   # token mean colour
        x = (torch.from_numpy(fr)[None].permute(0, 4, 1, 2, 3).float() / 255 - MEAN) / STD
        toks = {L: torch.nn.functional.normalize(v.float(), dim=-1).reshape(S, G * G, -1).numpy()
                for L, v in layer_tokens(enc, x, set(layers)).items()}
        # grouping: object-token pairs within a slot
        for sl in range(S):
            idx = np.flatnonzero(lab[sl] > 0)
            if len(idx) < 2:
                continue
            i, j = np.triu_indices(len(idx), 1)
            if len(i) > pairs_per_slot:
                k = rng.choice(len(i), pairs_per_slot, replace=False); i, j = i[k], j[k]
            ti, tj = idx[i], idx[j]
            same = lab[sl, ti] == lab[sl, tj]
            dist = np.maximum(abs(yy[ti] - yy[tj]), abs(xx[ti] - xx[tj]))
            ocol = {o: col[sl, lab[sl] == o].mean(0) for o in np.unique(lab[sl, idx])}
            cdiff = np.array([np.linalg.norm(ocol[lab[sl, p]] - ocol[lab[sl, q]]) for p, q in zip(ti, tj)])
            for L in layers:
                R["pairs"][L].append((toks[L][sl, ti] * toks[L][sl, tj]).sum(-1))
            R["pair_meta"].append(np.stack([same, dist, cdiff], 1)); R["pair_clip"].append(np.full(len(ti), n))
        # identity: object means at slot s and s + gap
        for sl in range(S - gap):
            objs = [o for o in np.unique(lab[sl]) if o > 0 and (lab[sl] == o).sum() >= 2
                    and (lab[sl + gap] == o).sum() >= 2]
            if len(objs) < 2:
                continue
            c0 = np.array([[yy[lab[sl] == o].mean(), xx[lab[sl] == o].mean()] for o in objs])
            c1 = np.array([[yy[lab[sl + gap] == o].mean(), xx[lab[sl + gap] == o].mean()] for o in objs])
            moved = np.abs(c1 - c0).max(1) >= 2
            for L in layers:
                m0 = np.stack([toks[L][sl, lab[sl] == o].mean(0) for o in objs])
                m1 = np.stack([toks[L][sl + gap, lab[sl + gap] == o].mean(0) for o in objs])
                R["ident"][L].append((m0 @ m1.T).argmax(1) == np.arange(len(objs)))
            R["ident_meta"].append(np.stack([np.full(len(objs), 1 / len(objs)), moved, np.full(len(objs), n)], 1))
        # figure-ground: up to 32 object and 32 background tokens from the last slot
        last = lab[-1]
        ob, bg = np.flatnonzero(last > 0), np.flatnonzero(last == 0)
        if len(ob) >= 4:
            ob = rng.choice(ob, min(32, len(ob)), replace=False); bg = rng.choice(bg, len(ob), replace=False)
            for L in layers:
                R["fig"][L].append(toks[L][-1, np.concatenate([ob, bg])])
            R["fig_y"].append(np.r_[np.ones(len(ob)), np.zeros(len(bg))]); R["fig_clip"].append(np.full(2 * len(ob), n))
        # per object in the last slot (>= 2 tokens): kind, size, 3D
        spec = {o["id"]: o for o in ep["objects"]}
        pos = {o["id"]: np.array(o["pos"]) for o in state["frames"][-1]["objects"]}
        cam = ep["scene"]["camera"]; cp = np.array(cam["pos"]); vd = np.array(cam["lookat"]) - cp; vd /= np.linalg.norm(vd)
        for o in np.unique(last[last > 0]):
            m = last == o
            if m.sum() < 2:
                continue
            sp = spec[o - 1]
            if sp["shape"] not in SHAPES:
                raise ValueError(f"unknown shape {sp['shape']}")
            if o - 1 not in pos:
                raise ValueError(f"object {o - 1} is in the last slot's labels but not in the last frame's state")
            p = pos[o - 1]
            R["obj_meta"].append([SHAPES.index(sp["shape"]), np.log(_volume(sp["shape"], sp["size"]) ** (1 / 3)),
                                  float((p - cp) @ vd), float(p[2] - _z_extent(sp["shape"], sp["size"])),
                                  yy[m].mean(), xx[m].mean(), np.log(m.sum()), n])
            for L in layers:
                R["obj"][L].append(toks[L][-1, m].mean(0))
        n += 1
        if n == n_clips:
            break
    if n < n_clips:
        raise ValueError(f"{data}: only {n} clips, need {n_clips}")
    cat = lambda v: np.concatenate(v) if v and np.ndim(v[0]) else np.array(v)  # noqa: E731
    out = {"n": n, "pair_clip": cat(R["pair_clip"]), "pair_meta": cat(R["pair_meta"]),
           "ident_meta": cat(R["ident_meta"]), "fig_y": cat(R["fig_y"]), "fig_clip": cat(R["fig_clip"]),
           "obj_meta": np.array(R["obj_meta"], dtype=float)}
    for L in layers:
        out[L] = {"pairs": cat(R["pairs"][L]), "ident": cat(R["ident"][L]), "fig": cat(R["fig"][L]),
                  "obj": np.stack(R["obj"][L])}
    return out


def score(raw, L, boot=0, seed=0):
    """Metrics at layer L; with boot > 0, also [lo, hi] 95% bootstrap CIs over clips."""
    n = raw["n"]; ntr = int(0.7 * n); d = raw[L]
    same, dist, cd = raw["pair_meta"][:, 0] > 0, raw["pair_meta"][:, 1], raw["pair_meta"][:, 2]
    hard = same | (cd <= np.quantile(cd[~same], 0.25))
    chance, moved, iclip = raw["ident_meta"][:, 0], raw["ident_meta"][:, 1] > 0, raw["ident_meta"][:, 2].astype(int)
    # probes: fit once on training clips, then score (and bootstrap) the test clips
    ftr = raw["fig_clip"] < ntr
    fig_p = LogisticRegression(C=0.1, max_iter=2000).fit(d["fig"][ftr], raw["fig_y"][ftr]).predict(d["fig"][~ftr])
    om = raw["obj_meta"]; oclip = om[:, 7].astype(int); otr = oclip < ntr
    kind_p = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[1e-3, 1e-2, 0.1, 1], max_iter=5000)).fit(
        d["obj"][otr], om[otr, 0]).predict(d["obj"][~otr])
    ridge = lambda X, y: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 11))).fit(  # noqa: E731
        X[otr], y[otr]).predict(X[~otr])
    size_p = ridge(d["obj"], om[:, 1])
    img = np.c_[om[:, 4:7], np.ones(len(om))]                      # image centroid y, x and log token area
    resid = {}
    for name, col in (("depth3d", 2), ("height3d", 3)):
        beta = np.linalg.lstsq(img[otr], om[otr, col], rcond=None)[0]
        r = om[:, col] - img @ beta
        resid[name] = (r[~otr], ridge(d["obj"], r))

    g_all = MatchedAUC(d["pairs"], same, dist)
    g_hard = MatchedAUC(d["pairs"][hard], same[hard], dist[hard])

    def metrics(wc):
        """wc[c] = how many times clip c is drawn."""
        wp, wi = wc[raw["pair_clip"]], wc[iclip]
        wf, wo = wc[raw["fig_clip"][~ftr]], wc[oclip[~otr]]
        r = {"grouping": g_all(wp), "grouping_hard": g_hard(wp[hard]),
             "identity": float((wi * d["ident"]).sum() / wi.sum()),
             "identity_moved": float((wi * d["ident"])[moved].sum() / wi[moved].sum()),
             "identity_chance": float((wi * chance).sum() / wi.sum()),
             "figure": _bal(raw["fig_y"][~ftr], fig_p, wf),
             "kind": _bal(om[~otr, 0], kind_p, wo),
             "size_r2": _r2(om[~otr, 1], size_p, wo)}
        for name, (y, p) in resid.items():
            r[f"{name}_r2"] = _r2(y, p, wo)
        return r

    out = metrics(np.ones(n))
    out["identity_moved_n"] = int(moved.sum()); out["objects_test"] = int((~otr).sum())
    if boot:
        rng = np.random.default_rng(seed)
        draws = [metrics(np.bincount(np.r_[rng.integers(0, ntr, ntr), rng.integers(ntr, n, n - ntr)], minlength=n)
                         .astype(float)) for _ in range(boot)]
        for k in draws[0]:
            v = np.array([dr[k] for dr in draws])
            out[f"{k}_ci"] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="/mnt/data/jepamath/v4b_probe")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8])
    ap.add_argument("--gap", type=int, default=4, help="identity: slots between the two views of an object")
    ap.add_argument("--pairs-per-slot", type=int, default=300)
    ap.add_argument("--boot", type=int, default=0, help="bootstrap resamples for 95%% CIs (0 = none)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("run with CUDA_VISIBLE_DEVICES=")
    torch.set_num_threads(40)
    enc, _t, _p, step = load_run(a.run, a.weights, device="cpu")
    enc.eval()
    if max(a.layers) > len(enc.blocks):
        raise ValueError(f"encoder has {len(enc.blocks)} layers")
    res = json.load(open(os.path.join(a.run, "args.json")))["res"]
    raw = collect(enc, res, a.data, a.n, a.layers, a.gap, a.pairs_per_slot)
    out = {"run": a.run, "step": step, "data": a.data, "n": a.n, "gap": a.gap, "boot": a.boot, "layers": {}}
    for L in a.layers:
        r = score(raw, L, a.boot)
        out["layers"][L] = r
        print(f"layer {L}: grouping AUC {r['grouping']:.3f} (hard {r['grouping_hard']:.3f})  identity "
              f"{r['identity'] * 100:.1f}% (moved {r['identity_moved'] * 100:.1f}%, n={r['identity_moved_n']}; "
              f"chance {r['identity_chance'] * 100:.1f}%)  figure {r['figure'] * 100:.1f}%  kind {r['kind'] * 100:.1f}%  "
              f"R2 size {r['size_r2']:.2f} depth {r['depth3d_r2']:.2f} "
              f"height {r['height3d_r2']:.2f}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
