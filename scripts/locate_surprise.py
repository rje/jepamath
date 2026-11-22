"""Where is the surprise? Track B prediction error inside the region a hidden change altered (CPU).

The possible clip and its impossible / control twin are pixel-identical up to the screen lowering, so the
predictor (context = frames before the screen lowers) makes ONE prediction for all three. After the reveal,
the twins differ from the possible clip only where an object was added, deleted, moved or swapped. The
token mask M is that pixel diff, so per seed and inside M the question is: which ending does the shared
prediction look like? For a hidden deletion, M is where the missing object should be:
    d_pos(M) < d_imp(M)  -> the prediction contains the object (it was kept in mind behind the screen)
    d_pos(M) > d_imp(M)  -> the prediction shows empty table there
Run on step-0 weights too: objects may be harder to predict than table whatever the model believes.

"snapshot" asks whether the model tracks the transfers it watched at all: a synthetic ending whose post-reveal
frames are the last frame before the screen rose (the box as last seen). M is where that differs from the true
possible ending (the transferred objects). d_twin < d_pos there means the prediction is the stale snapshot.

    CUDA_VISIBLE_DEVICES= python scripts/locate_surprise.py RUN_DIR --weights weights_030000.pt --out OUT
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from jepaeval.twins import _screen_times  # noqa: E402
from jepaeval.vjepa import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from jepatrain.reeval import load_run  # noqa: E402

FPS = 4.0
MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1)


def timeline(ep: dict, T: int, tb: int) -> tuple[int, int]:
    """(context frames, first post-reveal slot), the same rule as jepaeval.twins.reveal_on_triples."""
    t0, t1 = _screen_times(ep)
    ctx = int(t0 * FPS) // tb * tb
    first = next(j for j in range((T - ctx) // tb) if ctx + tb * (j + 1) - 1 >= math.ceil(t1 * FPS))
    return ctx, first


def token_mask(a: np.ndarray, b: np.ndarray, patch: int, thresh: int, min_frac: float) -> np.ndarray:
    """[T,H,W,3] uint8 post-reveal frames -> bool [grid*grid]: patches where a and b differ."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1).max(0) > thresh   # [H, W]
    g = d.shape[0] // patch
    frac = d.reshape(g, patch, g, patch).mean((1, 3))
    return (frac >= min_frac).reshape(-1)


@torch.no_grad()
def analyse(enc, tgt, pred, t: dict, args) -> dict:
    tb, G = enc.tubelet, enc.grid * enc.grid
    T = t["possible"].shape[0]
    ctx, first = timeline(t["pos_episode"], T, tb)
    for k in ("impossible", "control"):
        if not np.array_equal(t["possible"][:ctx], t[k][:ctx]):
            raise ValueError(f"{t['seed_id']}: {k} context differs from possible; the shared-prediction "
                             f"argument does not hold")
    post = slice(ctx + first * tb, T)
    raise_t = next(a["t_start"] for a in t["pos_episode"]["actions"] if a["kind"] == "screen_raise")
    f_pre = math.ceil(raise_t * FPS) - 1
    if f_pre < 0:
        raise ValueError(f"{t['seed_id']}: screen rises at t={raise_t}, no frame of the box before it")
    snap = t["possible"].copy()
    snap[post] = snap[f_pre]
    clips = {"possible": t["possible"], "impossible": t["impossible"], "control": t["control"], "snapshot": snap}
    x = torch.stack([torch.from_numpy(c) for c in clips.values()])
    x = (x.permute(0, 4, 1, 2, 3).float() / 255 - MEAN) / STD                       # [3, 3, T, H, W]
    n_ctx, N = ctx // tb * G, T // tb * G
    ar = lambda a, b: torch.arange(a, b).unsqueeze(0)
    p = pred(enc(x[:1, :, :ctx]), ar(0, n_ctx), ar(n_ctx, N)).float()[0]           # [Nt, D]
    h = tgt(x).float()
    h = F.layer_norm(h, (h.size(-1),))[:, n_ctx:]                                    # [3, Nt, D]
    S = (T - ctx) // tb
    err = (p[None] - h).abs().mean(-1).view(len(clips), S, G)[:, first:].mean(1).numpy()  # [4, G] post-reveal
    rec = {"seed_id": t["seed_id"], "imp_violation": t["imp_meta"]["violation"],
           "ctl_violation": t["ctl_meta"]["violation"], "op": t["meta"]["op"], "n0": int(t["meta"]["n0"]),
           "final": int(t["meta"]["final"]), "k": int(t["meta"]["k"])}
    for j, k in enumerate(list(clips)[1:], 1):
        M = token_mask(t["possible"][post], clips[k][post], enc.patch, args.thresh, args.min_frac)
        rec[k] = {"mask_tokens": int(M.sum())}
        if M.sum() == 0:
            continue
        rec[k].update(d_pos_in=float(err[0][M].mean()), d_twin_in=float(err[j][M].mean()),
                      d_pos_out=float(err[0][~M].mean()), d_twin_out=float(err[j][~M].mean()))
    return rec


def summarise(recs: list[dict]) -> dict:
    out = {}
    groups = [("imp=add", "impossible", lambda r: r["imp_violation"] == "add"),
              ("imp=delete", "impossible", lambda r: r["imp_violation"] == "delete"),
              ("ctl=move", "control", lambda r: r["ctl_violation"] == "move"),
              ("ctl=swap", "control", lambda r: r["ctl_violation"] == "swap")]
    for v in ("add", "delete"):
        for op in sorted({r["op"] for r in recs}):
            groups.append((f"imp={v},op={op}", "impossible",
                           lambda r, v=v, op=op: r["imp_violation"] == v and r["op"] == op))
    for op in sorted({r["op"] for r in recs}):
        groups.append((f"snapshot,op={op}", "snapshot", lambda r, op=op: r["op"] == op))
    for name, kind, f in groups:
        rs = [r[kind] for r in recs if f(r) and "d_pos_in" in r[kind]]
        if not rs:
            continue
        din = np.array([r["d_twin_in"] - r["d_pos_in"] for r in rs])
        dout = np.array([r["d_twin_out"] - r["d_pos_out"] for r in rs])
        out[name] = {"n": len(rs), "n_empty_mask": sum(1 for r in recs if f(r) and "d_pos_in" not in r[kind]),
                     "twin>pos_in": float((din > 0).mean() * 100), "diff_in": float(din.mean()),
                     "sem_in": float(din.std(ddof=1) / math.sqrt(len(din))),
                     "twin>pos_out": float((dout > 0).mean() * 100), "diff_out": float(dout.mean()),
                     "d_pos_in": float(np.mean([r["d_pos_in"] for r in rs])),
                     "d_twin_in": float(np.mean([r["d_twin_in"] for r in rs])),
                     "mask_tokens": float(np.mean([r["mask_tokens"] for r in rs]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--weights", nargs="+", required=True, help="evaluated for every run")
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--thresh", type=int, default=30, help="per-pixel max-channel diff (0-255) to count as changed")
    ap.add_argument("--min-frac", type=float, default=0.15, help="fraction of a patch's pixels that must change")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("run with CUDA_VISIBLE_DEVICES= : this is a CPU job")
    torch.set_num_threads(a.threads)
    # one process, held-out set loaded once (~23 GB): several of these at once ran the machine out of memory
    triples = torch.load(a.heldout, weights_only=False)[: a.n]
    os.makedirs(a.out, exist_ok=True)
    for run in a.runs:
        for w in a.weights:
            evaluate_one(run, w, triples, a)


def evaluate_one(run: str, weights: str, triples: list[dict], a) -> None:
    enc, tgt, pred, step = load_run(run, weights, device="cpu")
    for m in (enc, tgt, pred):
        m.eval()
    recs = []
    for i, t in enumerate(triples):
        recs.append(analyse(enc, tgt, pred, t, a))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(triples)}", flush=True)
    s = summarise(recs)
    name = f"{os.path.basename(run.rstrip('/'))}_{step:06d}"
    with open(os.path.join(a.out, f"locate_{name}.json"), "w") as f:
        json.dump({"run": run, "step": step, "args": vars(a), "summary": s, "records": recs}, f)
    print(f"{name}", flush=True)
    for g, v in s.items():
        print(f"  {g:<22} n={v['n']:4d} (empty {v['n_empty_mask']:3d}) mask {v['mask_tokens']:5.1f} tok | "
              f"twin>pos IN {v['twin>pos_in']:5.1f}% diff {v['diff_in']:+.4f}±{v['sem_in']:.4f} "
              f"| OUT {v['twin>pos_out']:5.1f}% {v['diff_out']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
