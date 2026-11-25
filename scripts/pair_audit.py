"""Pair-level audit of SupCon labels (S3 design, no training): what the contrastive loss is told about pairs of
tokens, scored against the renderer's per-token object labels (token_seg).

SupCon (jepatrain.train.supcon) takes every labelled token as an anchor; tokens with its label are positives and
every other token of the clip, labelled or not, is a negative. For anchors whose true object is o:
  fn_share  : of the anchor's same-object tokens, the share it is told are negatives (split / unrecalled tokens)
  fp_share  : of the anchor's positives, the share that are not its object (merges, ghosts)
  anchors   : labelled tokens, as a share of all object tokens
  fn_share_ignore_unlabelled : fn_share when unlabelled tokens of the clip are not used as negatives (only
              same-object tokens under another label remain: track id switches, split objects)
  fn_share_clean : also only same-slot tokens as same-clip negatives (positives still span the clip), so a track
              id switch across time no longer marks the object's earlier tokens as negatives
fp_share does not count pairs of two background tokens inside one group (a ghost with itself).
Pairs are counted over all tokens of a clip (SupCon samples tokens uniformly, which leaves these shares unchanged).
Label sources: render (token_seg), render+flip30, render+merge (jepatrain.train.corrupt_labels, the S0 tolerance
runs), motion (motion_seg), track (track_seg).

    python scripts/pair_audit.py --data /mnt/data/jepamath/v4b --n 300 --out OUT.json
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from jepagen.shards import load_sample  # noqa: E402
from jepatrain.data import downsample, motion_seg, token_seg, track_seg  # noqa: E402
from jepatrain.train import corrupt_labels  # noqa: E402


def fn_same_slot(g: np.ndarray, t: np.ndarray) -> float:
    """Same-object negatives when only labelled tokens of the anchor's own slot are negatives (g, t [S, G, G])."""
    out = 0
    for gs, ts in zip(g.astype(np.int64), t.astype(np.int64)):
        C = np.zeros((max(g.max(), 1) + 1, max(t.max(), 1) + 1), dtype=np.int64)
        np.add.at(C, (gs.ravel(), ts.ravel()), 1)
        obj = C[1:, 1:]
        out += (obj * (obj.sum(0)[None] - obj)).sum()
    return float(out)


def pair_counts(g: np.ndarray, t: np.ndarray) -> np.ndarray:
    """[tp, fn, fp, anchors_on_objects, object_tokens] pair counts for labels g against truth t (flat, 0 = none).
    Anchors are labelled tokens; a pair (anchor, other) is positive iff they share a label."""
    g_sh, t_sh = g, t
    g, t = g.astype(np.int64).ravel(), t.astype(np.int64).ravel()
    C = np.zeros((g.max() + 1, t.max() + 1), dtype=np.int64)
    np.add.at(C, (g, t), 1)
    Cg, Ct = C[1:], C[1:, :]                    # labelled groups only (anchors)
    T = C.sum(0)                                # tokens per true object (all labels)
    Gk = C.sum(1)[1:]                           # tokens per group
    obj = Cg[:, 1:]                             # anchors on true objects
    tp = (obj * (obj - 1)).sum()
    fn = (obj * (T[None, 1:] - obj)).sum()
    fp = (Cg * (Gk[:, None] - Cg)).sum()        # every anchor's positives that are a different true label
    fn_lab = (obj * (obj.sum(0)[None] - obj)).sum()   # same-object negatives among labelled tokens only
    return np.array([tp, fn, fp, obj.sum(), T[1:].sum(), fn_lab, fn_same_slot(g_sh, t_sh)], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--res", type=int, default=128); ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--tubelet", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    shards = sorted(glob.glob(os.path.join(a.data, "train", "possible", "*.tar")))
    if not shards:
        raise FileNotFoundError(f"no shards under {a.data}/train/possible")
    tot: dict[str, np.ndarray] = {}
    n = 0
    for s in wds.WebDataset(shards, shardshuffle=False, empty_check=False):
        ep, frames, seg, state = load_sample(s)
        fr = downsample(frames, a.res)
        t = token_seg(seg, a.res, a.patch, a.tubelet)
        tl = torch.from_numpy(t.reshape(1, -1).astype(np.int64))
        src = {"render": t,
               "render+flip30": corrupt_labels(tl, 0.3, False).numpy().reshape(t.shape),
               "render+merge": corrupt_labels(tl, 0.0, True).numpy().reshape(t.shape),
               "motion": motion_seg(fr, a.patch, a.tubelet),
               "track": track_seg(fr, a.patch, a.tubelet)}
        for k, g in src.items():
            tot[k] = tot.get(k, 0) + pair_counts(g, t)
        n += 1
        if n == a.n:
            break
    if n < a.n:
        raise ValueError(f"only {n} clips, need {a.n}")
    res = {"data": a.data, "n": n}
    for k, (tp, fn, fp, anc, objt, fn_lab, fn_slot) in tot.items():
        res[k] = {"fn_share": fn / (tp + fn), "fp_share": fp / (tp + fp) if tp + fp else None,
                  "anchors": anc / objt, "fn_share_ignore_unlabelled": fn_lab / (tp + fn_lab),
                  "fn_share_clean": fn_slot / (tp + fn_slot)}
        print(f"{k:14s} fn_share {res[k]['fn_share']:.3f}  fp_share {res[k]['fp_share']:.3f}  "
              f"anchors {res[k]['anchors']:.3f}  fn_share if unlabelled ignored {res[k]['fn_share_ignore_unlabelled']:.3f}  "
              f"clean (also same-slot negatives only) {res[k]['fn_share_clean']:.3f}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
