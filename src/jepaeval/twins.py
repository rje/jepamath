"""Surprise on jepagen twin triples: possible / impossible (count changed) / control (count kept).

For each seed we compare clips that are pixel-identical until the hidden violation:
  imp>pos  : surprised by an impossible count (number sense, or any visual change)
  ctl>pos  : surprised by a count-preserving change (visual-change baseline)
  imp>ctl  : more surprised by the count change than by the matched visual change (number-specific)
"""

from __future__ import annotations

import glob
import io
import json
import math
import os

import numpy as np
import pyarrow.parquet as pq
import torch
import webdataset as wds

from .vjepa import VJepaSurprise


def _stream(paths):
    for s in wds.WebDataset(paths, shardshuffle=False, empty_check=False):
        z = np.load(io.BytesIO(s["clip.npz"]))
        yield s["__key__"], z["frames"], json.loads(s["episode.json"])


def load_triples(data: str, split: str, n: int, transform=None) -> list[dict]:
    """First n seeds that have possible, impossible and control clips. `transform` (frames -> frames) is
    applied as each clip is read, e.g. to downsample so thousands of triples fit in memory."""
    transform = transform or (lambda f: f)
    idx = pq.read_table(os.path.join(data, f"index-{split}.parquet")).to_pandas()
    ctl = idx[idx.twin_kind == "control"].set_index("twin_of")
    imp = idx[idx.twin_kind == "impossible"].set_index("twin_of")
    pos = idx[idx.twin_kind == "possible"].set_index("episode_id")
    shard = lambda kind: sorted(glob.glob(os.path.join(data, split, kind, "*.tar")))
    want: dict[str, dict] = {}
    for key, frames, ep in _stream(shard("impossible")):
        base = ep["twin_of"]
        if base not in ctl.index:
            continue
        want[base] = {"seed_id": base, "impossible": transform(frames), "meta": pos.loc[base].to_dict(),
                      "imp_meta": imp.loc[base].to_dict(), "ctl_meta": ctl.loc[base].to_dict()}
        if len(want) == n:
            break
    if len(want) < n:
        raise RuntimeError(f"only {len(want)} triples found, wanted {n}")
    for kind in ("possible", "control"):
        found = 0
        for key, frames, ep in _stream(shard(kind)):
            base = key if kind == "possible" else ep["twin_of"]
            if base in want:
                want[base][kind] = transform(frames)
                if kind == "possible":
                    want[base]["pos_episode"] = ep
                found += 1
                if found == n:
                    break
        if found < n:
            raise RuntimeError(f"found {found}/{n} {kind} clips")
    return list(want.values())


def _acc(a: torch.Tensor, b: torch.Tensor) -> dict:
    """a, b: [n, C, W]; fraction (in %) where a is more surprising than b, per context, by mean and max."""
    return {"avg": ((a.mean(-1) > b.mean(-1)).float().mean(0) * 100).tolist(),
            "max": ((a.max(-1).values > b.max(-1).values).float().mean(0) * 100).tolist()}


def evaluate(data: str, split: str, sj: VJepaSurprise, n: int, context_lengths=(2, 4, 6, 8, 10), log=print) -> dict:
    triples = load_triples(data, split, n)
    L = {k: [] for k in ("possible", "impossible", "control")}
    for i, t in enumerate(triples):
        for k in L:
            L[k].append(sj.window_losses(sj.preprocess(t[k]), context_lengths))
        if (i + 1) % 25 == 0:
            log(f"[twins] {i + 1}/{len(triples)} triples, guard {sj.guard.summary()}")
    P, I, C = (torch.stack(L[k]) for k in ("possible", "impossible", "control"))
    meta = [t["meta"] for t in triples]
    vk = np.array([t["imp_meta"]["violation"] for t in triples])
    ck = np.array([t["ctl_meta"]["violation"] for t in triples])
    final = np.array([m["final"] for m in meta])
    out = {"n": len(triples), "context_lengths": list(context_lengths),
           "imp>pos": _acc(I, P), "ctl>pos": _acc(C, P), "imp>ctl": _acc(I, C), "by": {}}
    for name, mask in [("imp=add", vk == "add"), ("imp=delete", vk == "delete"),
                       ("ctl=move", ck == "move"), ("ctl=swap", ck == "swap"),
                       ("final<=3", final <= 3), ("final>=6", final >= 6)]:
        m = torch.from_numpy(mask)
        if m.sum() == 0:
            continue
        out["by"][name] = {"n": int(m.sum()), "imp>pos": _acc(I[m], P[m]), "ctl>pos": _acc(C[m], P[m]),
                           "imp>ctl": _acc(I[m], C[m])}
    raw = {"possible": P, "impossible": I, "control": C, "seed_ids": [t["seed_id"] for t in triples],
           "imp_violation": vk.tolist(), "ctl_violation": ck.tolist(), "final": final.tolist(),
           "n0": [m["n0"] for m in meta], "op": [m["op"] for m in meta]}
    return {"metrics": out, "raw": raw}


def _screen_times(ep: dict) -> tuple[float, float]:
    a = [x for x in ep["actions"] if x.get("kind") == "screen_lower"]
    if len(a) != 1:
        raise ValueError(f"{ep['episode_id']}: expected one screen_lower, found {len(a)}")
    return a[0]["t_start"], a[0]["t_end"]


def _breakdown(I, P, C, groups):
    out = {"imp>pos": _pct(I, P), "ctl>pos": _pct(C, P), "imp>ctl": _pct(I, C), "by": {}}
    for name, mask in groups:
        m = torch.from_numpy(mask)
        if m.sum():
            out["by"][name] = {"n": int(m.sum()), "imp>pos": _pct(I[m], P[m]), "ctl>pos": _pct(C[m], P[m]),
                               "imp>ctl": _pct(I[m], C[m])}
    return out


def _pct(a, b):
    d = a - b
    return {"acc": float((d > 0).float().mean() * 100), "diff_mean": float(d.mean()),
            "diff_sem": float(d.std() / math.sqrt(len(d)))}


def evaluate_reveal(data: str, split: str, sj: VJepaSurprise, n: int, fps: float = 4.0, log=print) -> dict:
    return reveal_on_triples(load_triples(data, split, n), sj, fps, log)


def reveal_on_triples(triples: list[dict], sj, fps: float = 4.0, log=print) -> dict:
    """Reveal-locked surprise: one window spanning the whole clip (subsampled to the model's max window),
    context = every frame before the screen starts lowering (so the starting count, the visible transfers
    and the hidden-interval are all in context), score = mean surprise on the time slots after the screen
    is fully down (the count is visible). The twins share this timeline, so the possible clip's is used."""
    S = {k: [] for k in ("possible", "impossible", "control")}
    post, ctxs = [], []
    for i, t in enumerate(triples):
        T = t["possible"].shape[0]
        step = 1 if sj.max_window is None else T // sj.max_window
        if sj.max_window is not None and T % sj.max_window:
            raise ValueError(f"clip of {T} frames does not subsample evenly to {sj.max_window}")
        f = fps / step
        t0, t1 = _screen_times(t["pos_episode"])
        tb = sj.tubelet
        ctx = int(t0 * f) // tb * tb
        if ctx < tb:
            raise ValueError(f"{t['seed_id']}: screen lowers at {t0:.2f}s, no context")
        n_slots = (T // step - ctx) // tb
        first = next((j for j in range(n_slots) if ctx + tb * (j + 1) - 1 >= math.ceil(t1 * f)), None)
        if first is None:
            raise ValueError(f"{t['seed_id']}: no time slot after the reveal ends at {t1:.2f}s")
        for k in S:
            S[k].append(sj.slot_losses(sj.preprocess(t[k][::step]), ctx))
        post.append(first); ctxs.append(ctx)
        if (i + 1) % 25 == 0:
            log(f"[reveal] {i + 1}/{len(triples)} triples, guard {sj.guard.summary()}")
    score = {k: torch.stack([v[j:].mean() for v, j in zip(S[k], post)]) for k in S}
    whole = {k: torch.stack([v.mean() for v in S[k]]) for k in S}
    meta = [t["meta"] for t in triples]
    vk = np.array([t["imp_meta"]["violation"] for t in triples])
    ck = np.array([t["ctl_meta"]["violation"] for t in triples])
    final = np.array([m["final"] for m in meta])
    groups = [("imp=add", vk == "add"), ("imp=delete", vk == "delete"), ("ctl=move", ck == "move"),
              ("ctl=swap", ck == "swap"), ("final<=3", final <= 3), ("final>=6", final >= 6)]
    out = {"n": len(triples), "post_reveal": _breakdown(score["impossible"], score["possible"], score["control"], groups),
           "whole_target": _breakdown(whole["impossible"], whole["possible"], whole["control"], groups),
           "ctx_frames_median": float(np.median(ctxs))}
    raw = {"slots": S, "first_post_slot": post, "ctx": ctxs, "seed_ids": [t["seed_id"] for t in triples],
           "imp_violation": vk.tolist(), "ctl_violation": ck.tolist(), "final": final.tolist(),
           "n0": [m["n0"] for m in meta], "op": [m["op"] for m in meta]}
    return {"metrics": out, "raw": raw}
