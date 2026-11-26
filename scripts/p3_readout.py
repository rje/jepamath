"""P3: where does the C9 readout fail? (research log 27 Nov, pre-registered; Astra's suggestion)

On frozen C9 blocks, p = d(2-object ending) - d(3-object ending) under four variants, scored with C9's block endpoints and
shape-stratified bootstrap:
  l1          tokenwise mean L1 (C9's native readout; must reproduce c9_score)
  l2          tokenwise mean squared error (the training loss)
  pool_l1     mean over common tokens within each slot, then L1
  revealed_l1 native L1 with the source's ACTUAL revealed ending (target encoder, unmasked frames) in place of the prediction

    CUDA_VISIBLE_DEVICES= python scripts/p3_readout.py --manifest C9.json --out OUT.json --run R --weights W --name N [...]
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
from c9_score import block_endpoints, bootstrap, sha  # noqa: E402
from locate_surprise import MEAN, STD  # noqa: E402

TUB = 4
VARIANTS = ("l1", "l2", "pool_l1", "revealed_l1")


def prep(fr):
    x = torch.from_numpy(fr).permute(0, 4, 1, 2, 3).float() / 255
    return (x - MEAN) / STD


@torch.no_grad()
def prefs(xs_masked, xs_full, xd, ctx, first, common, enc, tgt, pred) -> dict[str, np.ndarray]:
    a, full, b = prep(xs_masked), prep(xs_full), prep(xd)
    G = enc.grid * enc.grid
    slots = a.shape[2] // TUB
    s0 = ctx // TUB + first
    tok = torch.from_numpy(np.asarray(common))
    keep = torch.cat([s * G + tok for s in range(s0, slots)])
    n_ctx, N = ctx // TUB * G, slots * G
    n = len(xs_masked)
    z = enc(a[:, :, :ctx])
    P = pred(z, torch.arange(n_ctx).unsqueeze(0).expand(n, -1),
             torch.arange(n_ctx, N).unsqueeze(0).expand(n, -1)).float()[:, keep - n_ctx]
    ln = lambda h: F.layer_norm(h, (h.size(-1),))
    T = ln(tgt(b, keep.unsqueeze(0).expand(len(xd), -1)).float())
    R = ln(tgt(full, keep.unsqueeze(0).expand(n, -1)).float())
    nb = len(tok)
    pool = lambda X: X.reshape(X.shape[0], -1, nb, X.shape[-1]).mean(2)        # [n, slots, D]
    D = {"l1": (P[:, None] - T[None]).abs().mean((2, 3)),
         "l2": ((P[:, None] - T[None]) ** 2).mean((2, 3)),
         "pool_l1": (pool(P)[:, None] - pool(T)[None]).abs().mean((2, 3)),
         "revealed_l1": (R[:, None] - T[None]).abs().mean((2, 3))}
    return {k: (M[:, 0] - M[:, 1]).numpy() for k, M in D.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--weights", action="append", default=[])
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--boot", type=int, default=10_000)
    ap.add_argument("--limit", type=int, default=0, help="smoke test only")
    a = ap.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("run with CUDA_VISIBLE_DEVICES=")
    if not (len(a.run) == len(a.weights) == len(a.name)):
        raise SystemExit("--run/--weights/--name must pair up")
    man = json.load(open(a.manifest))
    for d_, h in man["items_sha256"].items():
        if sha(os.path.join(d_, "items.jsonl")) != h:
            raise SystemExit(f"{d_}/items.jsonl changed since the freeze")
    srec = {json.loads(l)["seed"]: json.loads(l) for l in open(os.path.join(man["sources_dir"], "items.jsonl"))}
    blocks = man["blocks"][: a.limit] if a.limit else man["blocks"]
    strata = np.array([b["shape"] for b in blocks])
    torch.set_num_threads(16)
    per, res = {}, {"manifest": a.manifest, "n_blocks": len(blocks), "models": {}}
    for run, w, name in zip(a.run, a.weights, a.name):
        enc, tgt, pred, step = load_model(run, w)
        enc.eval(); tgt.eval(); pred.eval()
        rows = {v: [] for v in VARIANTS}
        for b in blocks:
            s = srec[b["source"]]
            zs = np.load(os.path.join(man["sources_dir"], f"source_{b['source']:09d}.npz"))
            xs = mask_hopper(zs["frames"], zs["hopper_px"], zs["hopper_ring"], s["ctx"])
            P = {v: {} for v in VARIANTS}
            for lay in ("rev", "fwd"):
                zd = np.load(os.path.join(man[f"{lay}_dir"], f"donor_{b[lay]['donor']:09d}.npz"))
                pr = prefs(xs, zs["frames"], zd["frames"], s["ctx"], s["first_post_slot"], np.array(b[lay]["common"]),
                           enc, tgt, pred)
                for v in VARIANTS:
                    P[v][lay] = dict(zip(s["names"], map(float, pr[v])))
            for v in VARIANTS:
                rows[v].append(block_endpoints(P[v]))
        for v in VARIANTS:
            per[f"{name}:{v}"] = rows[v]
        res["models"][name] = {"run": run, "weights": w, "step": step}
        for v in VARIANTS:
            print(f"[{name}:{v}] " + "  ".join(f"{k} {np.mean([r[k] for r in rows[v]]):.3f}"
                                               for k in ("conflict:orig", "pair_area:conflict", "pair_volume:conflict")),
                  flush=True)
    pairs = [(f"{m}:{v}", f"{m}:l1") for m in res["models"] for v in VARIANTS if v != "l1"]
    res["bootstrap"] = bootstrap(per, strata, a.boot, pairs)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
