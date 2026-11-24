"""Object sense in a frozen encoder, before any counting: are objects there as units?

On fresh probe clips (whole clip, all slots, the online encoder), scored against the renderer's per-token object
labels (jepatrain.data.token_seg; claw and screen count as background). Readout-free except figure-ground.
  grouping  : over token pairs in one slot, both on objects, AUC of cosine similarity for "same object" vs
              "different objects", matched for distance (AUC within Chebyshev-distance bins, then averaged by
              pair count). `grouping_hard`: only different-object pairs whose mean pixel colours are close
              (bottom quartile), so colour alone cannot separate them.
  identity  : each object's mean token at slot s vs slot s + gap; accuracy of matching every object to itself
              among the objects visible in both (chance = mean 1/K). `identity_moved`: only objects whose centre
              moved >= 2 tokens, so position alone cannot match them.
  figure    : logistic probe, object vs background token, balanced accuracy on held-out clips.
Each is reported at the embedding (layer 0: appearance + position, the no-learning baseline), a middle layer
and the output.

    CUDA_VISIBLE_DEVICES= python scripts/object_sense.py RUN --weights W --out OUT.json
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from jepagen.shards import load_sample  # noqa: E402
from jepatrain.data import downsample, token_seg  # noqa: E402
from jepatrain.reeval import load_run  # noqa: E402
from locate_surprise import MEAN, STD  # noqa: E402

DIST_BINS = (1, 2, 3, 5, 9, 99)          # Chebyshev distance bins on the token grid: [1,2), [2,3), [3,5), ...


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


def auc_matched(sim, same, dist):
    """AUC of sim for same vs different, within distance bins, weighted by pairs in bins that have both."""
    tot, w = 0.0, 0
    for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:]):
        k = (dist >= lo) & (dist < hi)
        if same[k].any() and (~same[k]).any():
            tot += roc_auc_score(same[k], sim[k]) * k.sum(); w += k.sum()
    return tot / w if w else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="/mnt/data/jepamath/v4b_probe")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8])
    ap.add_argument("--gap", type=int, default=4, help="identity: slots between the two views of an object")
    ap.add_argument("--pairs-per-slot", type=int, default=300)
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
    G, S, P, tb = enc.grid, enc.slots, enc.patch, enc.tubelet
    rng = np.random.default_rng(0)
    pairs = {L: [] for L in a.layers}            # (sim, same, dist, colour diff) per pair
    ident = {L: [] for L in a.layers}            # (correct, 1/K, moved) per object view
    figX = {L: [] for L in a.layers}; figy, figclip = [], []
    yy, xx = np.divmod(np.arange(G * G), G)
    n = 0
    for s in wds.WebDataset(sorted(glob.glob(os.path.join(a.data, "train", "possible", "*.tar"))),
                            shardshuffle=False, empty_check=False):
        _ep, frames, seg, _state = load_sample(s)
        fr = downsample(frames, res)
        if fr.shape[0] != S * tb:
            raise ValueError(f"clip has {fr.shape[0]} frames, encoder wants {S * tb}")
        lab = token_seg(seg, res, P, tb).reshape(S, G * G).astype(int)
        col = fr.reshape(S, tb, G, P, G, P, 3).mean((1, 3, 5)).reshape(S, G * G, 3)   # token mean colour
        x = (torch.from_numpy(fr)[None].permute(0, 4, 1, 2, 3).float() / 255 - MEAN) / STD
        toks = {L: torch.nn.functional.normalize(v.float(), dim=-1).reshape(S, G * G, -1).numpy()
                for L, v in layer_tokens(enc, x, set(a.layers)).items()}
        # grouping: object-token pairs within a slot
        for sl in range(S):
            idx = np.flatnonzero(lab[sl] > 0)
            if len(idx) < 2:
                continue
            i, j = np.triu_indices(len(idx), 1)
            if len(i) > a.pairs_per_slot:
                k = rng.choice(len(i), a.pairs_per_slot, replace=False); i, j = i[k], j[k]
            ti, tj = idx[i], idx[j]
            same = lab[sl, ti] == lab[sl, tj]
            dist = np.maximum(abs(yy[ti] - yy[tj]), abs(xx[ti] - xx[tj]))
            ocol = {o: col[sl, lab[sl] == o].mean(0) for o in np.unique(lab[sl, idx])}
            cdiff = np.array([np.linalg.norm(ocol[lab[sl, p]] - ocol[lab[sl, q]]) for p, q in zip(ti, tj)])
            for L in a.layers:
                sim = (toks[L][sl, ti] * toks[L][sl, tj]).sum(-1)
                pairs[L].append(np.stack([sim, same, dist, cdiff], 1))
        # identity: object means at slot s and s + gap
        for sl in range(S - a.gap):
            objs = [o for o in np.unique(lab[sl]) if o > 0 and (lab[sl] == o).sum() >= 2
                    and (lab[sl + a.gap] == o).sum() >= 2]
            if len(objs) < 2:
                continue
            c0 = np.array([[yy[lab[sl] == o].mean(), xx[lab[sl] == o].mean()] for o in objs])
            c1 = np.array([[yy[lab[sl + a.gap] == o].mean(), xx[lab[sl + a.gap] == o].mean()] for o in objs])
            moved = np.abs(c1 - c0).max(1) >= 2
            for L in a.layers:
                m0 = np.stack([toks[L][sl, lab[sl] == o].mean(0) for o in objs])
                m1 = np.stack([toks[L][sl + a.gap, lab[sl + a.gap] == o].mean(0) for o in objs])
                hit = (m0 @ m1.T).argmax(1) == np.arange(len(objs))
                ident[L] += [(h, 1 / len(objs), mv) for h, mv in zip(hit, moved)]
        # figure-ground: up to 32 object and 32 background tokens from the last slot
        ob, bg = np.flatnonzero(lab[-1] > 0), np.flatnonzero(lab[-1] == 0)
        if len(ob) >= 4:
            ob = rng.choice(ob, min(32, len(ob)), replace=False); bg = rng.choice(bg, len(ob), replace=False)
            for L in a.layers:
                figX[L].append(toks[L][-1, np.concatenate([ob, bg])])
            figy.append(np.r_[np.ones(len(ob)), np.zeros(len(bg))]); figclip.append(np.full(2 * len(ob), n))
        n += 1
        if n == a.n:
            break
    if n < a.n:
        raise ValueError(f"{a.data}: only {n} clips, need {a.n}")
    figy, figclip = np.concatenate(figy), np.concatenate(figclip)
    tr = figclip < int(0.7 * a.n)
    out = {"run": a.run, "step": step, "data": a.data, "n": a.n, "gap": a.gap, "layers": {}}
    for L in a.layers:
        pr = np.concatenate(pairs[L]); sim, same, dist, cd = pr[:, 0], pr[:, 1] > 0, pr[:, 2], pr[:, 3]
        q = np.quantile(cd[~same], 0.25)
        hard = same | (cd <= q)
        idt = np.array(ident[L], dtype=float)
        X = np.concatenate(figX[L])
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(X[tr], figy[tr])
        p = clf.predict(X[~tr]); t = figy[~tr]
        r = {"grouping": auc_matched(sim, same, dist), "grouping_hard": auc_matched(sim[hard], same[hard], dist[hard]),
             "identity": idt[:, 0].mean(), "identity_chance": idt[:, 1].mean(),
             "identity_moved": idt[idt[:, 2] > 0, 0].mean(), "identity_moved_n": int((idt[:, 2] > 0).sum()),
             "figure": 0.5 * ((p[t == 1] == 1).mean() + (p[t == 0] == 0).mean())}
        out["layers"][L] = r
        print(f"layer {L}: grouping AUC {r['grouping']:.3f} (hard {r['grouping_hard']:.3f})  identity "
              f"{r['identity'] * 100:.1f}% (moved {r['identity_moved'] * 100:.1f}%, n={r['identity_moved_n']}; "
              f"chance {r['identity_chance'] * 100:.1f}%)  figure {r['figure'] * 100:.1f}%")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
