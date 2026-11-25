"""Offline quality of a label-free grouper against the renderer's per-token object labels (S2, no training).

On raw clips (frames + seg + state), downsampled to the training resolution as prep_cloud does, run a grouper
(jepatrain.data.motion_seg or track_seg; pixels only) and score its token groups against token_seg:
  recall       : object tokens that get some group
  precision    : grouped tokens that lie on an object (1 - precision = ghosts and claw/screen tokens)
  purity       : grouped object tokens whose group's majority object is their own object (merges lower it)
  consistency  : grouped object tokens whose group is their object's majority group over the clip (an object that
                 changes id, e.g. after a stop or an occlusion, lowers it)
  recall_rest  : recall on tokens of objects that moved earlier in the clip and are now at rest (persistence:
                 a motion-only grouper loses them)
  recall_never : recall on tokens of objects that never move in the clip (no grouper from motion can get these)
A slot counts an object as moving if its position changes by more than --move-thresh m inside that tubelet.

    python scripts/grouper_eval.py --grouper motion --n 300
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import webdataset as wds

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from jepagen.shards import load_sample  # noqa: E402
from jepatrain.data import downsample, motion_seg, token_seg, track_seg  # noqa: E402

GROUPERS = {"motion": motion_seg, "track": track_seg}


def motion_state(state: dict, n_slots: int, tubelet: int, thresh: float) -> dict[int, np.ndarray]:
    """object id -> int [S]: 0 never moved yet, 1 moving in this slot, 2 at rest after having moved."""
    out = {}
    ids = {o["id"] for f in state["frames"] for o in f["objects"]}
    for oid in ids:
        pos = np.array([next((o["pos"] for o in f["objects"] if o["id"] == oid), [np.nan] * 3)
                        for f in state["frames"]])
        st = np.zeros(n_slots, dtype=int)
        moved = False
        for s in range(n_slots):
            p = pos[s * tubelet:(s + 1) * tubelet]
            p = p[~np.isnan(p).any(1)]
            mv = len(p) > 1 and float(np.linalg.norm(p - p[0], axis=1).max()) > thresh
            moved |= mv
            st[s] = 1 if mv else (2 if moved else 0)
        out[oid] = st
    return out


def score_clip(g: np.ndarray, t: np.ndarray, ms: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """Counts (numerator, denominator) per metric for one clip: g, t uint8 [S, G, G] (group, truth object id + 1)."""
    obj, grp = t > 0, g > 0
    c = {"recall": [(obj & grp).sum(), obj.sum()], "precision": [(obj & grp).sum(), grp.sum()]}
    both = obj & grp
    maj_obj = {k: np.bincount(t[(g == k) & obj]).argmax() for k in np.unique(g[both])}
    maj_grp = {k: np.bincount(g[(t == k) & grp]).argmax() for k in np.unique(t[both])}
    c["purity"] = [sum((maj_obj[k] == t[(g == k) & obj]).sum() for k in maj_obj), both.sum()]
    c["consistency"] = [sum((maj_grp[k] == g[(t == k) & grp]).sum() for k in maj_grp), both.sum()]
    for name, code in (("recall_rest", 2), ("recall_never", 0), ("recall_moving", 1)):
        sel = np.zeros_like(obj)
        for oid, st in ms.items():
            sel |= (t == oid + 1) & (st == code)[:, None, None]
        c[name] = [(sel & grp).sum(), sel.sum()]
    return {k: np.array(v, dtype=float) for k, v in c.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grouper", choices=sorted(GROUPERS), required=True)
    ap.add_argument("--data", default="/mnt/data/jepamath/v5")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--res", type=int, default=128); ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--tubelet", type=int, default=4)
    ap.add_argument("--move-thresh", type=float, default=0.005)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    shards = sorted(glob.glob(os.path.join(a.data, "train", "possible", "*.tar")))
    if not shards:
        raise FileNotFoundError(f"no shards under {a.data}/train/possible")
    tot, n = {}, 0
    for s in wds.WebDataset(shards, shardshuffle=False, empty_check=False):
        ep, frames, seg, state = load_sample(s)
        fr = downsample(frames, a.res)
        g = GROUPERS[a.grouper](fr, a.patch, a.tubelet)
        t = token_seg(seg, a.res, a.patch, a.tubelet)
        for k, v in score_clip(g, t, motion_state(state, t.shape[0], a.tubelet, a.move_thresh)).items():
            tot[k] = tot.get(k, 0) + v
        n += 1
        if n == a.n:
            break
    if n < a.n:
        raise ValueError(f"only {n} clips, need {a.n}")
    res = {k: float(v[0] / v[1]) if v[1] else None for k, v in tot.items()}
    res.update({f"{k}_tokens": int(v[1]) for k, v in tot.items()}, grouper=a.grouper, n=n)
    print(f"{a.grouper} ({n} clips): " + "  ".join(f"{k} {res[k]:.3f}" for k in tot if res[k] is not None))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
