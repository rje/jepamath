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


def load_triples(data: str, split: str, n: int) -> list[dict]:
    """First n seeds that have possible, impossible and control clips."""
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
        want[base] = {"seed_id": base, "impossible": frames, "meta": pos.loc[base].to_dict(),
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
                want[base][kind] = frames
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
