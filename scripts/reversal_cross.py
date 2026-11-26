"""Cross-scene history reversal (C4; research log 26 Nov). Each scene A is scored against a DONOR scene B's
four endings, so identity, layout and drop spots no longer match A's history; only the count can.

  - Pairing: the donor B (B != A) has the same ctx and first post-reveal slot, and the highest box-token IoU with
    A. IoU must be >= --min-iou; scenes with no such donor are dropped and counted.
  - Score: P_h from A's history h, with the hopper masked in context (the H-C2 arm). T_e from B's ending with
    final count e (target encoder on B's post-reveal tokens only). Both use the tokens in A's and B's shared box
    region. Ending order is by final count in both scenes (reversal_items saves histories sorted by final), so
    the diagonal is the count-matched ending.
Endpoints, bootstrap and oracles come from scripts/reversal_score.py. An oracle compares A's history h with B's
history e on the information it reads.

    python scripts/reversal_cross.py --items DIR --out OUT.json --run RUN --weights W --name NAME [...]
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
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from locate_surprise import MEAN, STD  # noqa: E402
from reversal_score import ORACLES, TUB, endpoints, finals, load_items, mask_hopper, summarize  # noqa: E402


def pair(items: list[dict], min_iou: float) -> tuple[list[tuple[int, int, np.ndarray]], int]:
    """[(a, b, shared box mask)] and the number of scenes dropped."""
    out, dropped = [], 0
    for a, A in enumerate(items):
        best, bi = -1.0, None
        for b, B in enumerate(items):
            if b == a or B["ctx"] != A["ctx"] or B["first_post_slot"] != A["first_post_slot"]:
                continue
            inter = (A["box"] & B["box"]).sum()
            iou = inter / max(1, (A["box"] | B["box"]).sum())
            if iou > best:
                best, bi = iou, b
        if bi is None or best < min_iou:
            dropped += 1
            continue
        out.append((a, bi, A["box"] & items[bi]["box"]))
    return out, dropped


def oracle_cross(A, B, key) -> np.ndarray:
    if key is None:
        fin = finals(B)
        return np.tile(np.abs(fin - A["n0"]).astype(float), (4, 1))
    ia = [key(A["meta"][h], A["n0"]) for h in A["histories"]]
    ib = [key(B["meta"][h], B["n0"]) for h in B["histories"]]
    return np.array([[0.0 if ia[h] == ib[e] else 1.0 for e in range(4)] for h in range(4)])


@torch.no_grad()
def cross_matrix(A, B, box, enc, tgt, pred) -> np.ndarray:
    def prep(fr):
        x = torch.from_numpy(fr).permute(0, 4, 1, 2, 3).float() / 255
        return (x - MEAN) / STD
    xa, xb = prep(mask_hopper(A)), prep(B["frames"])
    G = enc.grid * enc.grid
    ctx, first = A["ctx"], A["first_post_slot"]
    slots = xa.shape[2] // TUB
    s0 = ctx // TUB + first
    tok = torch.from_numpy(np.flatnonzero(box.ravel()))
    keep = torch.cat([s * G + tok for s in range(s0, slots)])
    n_ctx, N = ctx // TUB * G, slots * G
    z = enc(xa[:, :, :ctx])
    P = pred(z, torch.arange(n_ctx).unsqueeze(0).expand(4, -1),
             torch.arange(n_ctx, N).unsqueeze(0).expand(4, -1)).float()[:, keep - n_ctx]
    h = tgt(xb, keep.unsqueeze(0).expand(4, -1)).float()
    T = F.layer_norm(h, (h.size(-1),))
    return (P[:, None] - T[None]).abs().mean((2, 3)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-iou", type=float, default=0.6)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--weights", action="append", default=[])
    ap.add_argument("--name", action="append", default=[])
    a = ap.parse_args()
    if not a.run or not (len(a.run) == len(a.weights) == len(a.name)):
        raise SystemExit("--run/--weights/--name must pair up")
    items = load_items(a.items, 200, 905000)
    pairs, dropped = pair(items, a.min_iou)
    As = [items[i] for i, _, _ in pairs]
    ious = [float((items[i]["box"] & items[j]["box"]).sum() / (items[i]["box"] | items[j]["box"]).sum()) for i, j, _ in pairs]
    res = {"items": a.items, "n_pairs": len(pairs), "dropped": dropped, "iou_mean": float(np.mean(ious)),
           "n0_counts": {n: sum(x["n0"] == n for x in As) for n in (1, 2)}, "baselines": {}, "models": {}}
    print(f"{len(pairs)} pairs, {dropped} dropped, mean IoU {res['iou_mean']:.2f}, n0 {res['n0_counts']}", flush=True)
    for k, key in ORACLES.items():
        res["baselines"][k] = summarize(endpoints([oracle_cross(items[i], items[j], key) for i, j, _ in pairs], As,
                                                  ties="uniform"))
        print(f"[baseline] {k:20s} ident {res['baselines'][k]['ident']['est']:.3f}", flush=True)
    from jepatrain.reeval import load_run
    torch.set_num_threads(16)
    for run, w, name in zip(a.run, a.weights, a.name):
        enc, tgt, pred, step = load_run(run, w, device="cpu")
        enc.eval(); tgt.eval(); pred.eval()
        mats = [cross_matrix(items[i], items[j], box, enc, tgt, pred) for i, j, box in pairs]
        r = summarize(endpoints(mats, As))
        r["run"], r["weights"], r["step"] = run, w, step
        res["models"][name] = r
        print(f"[{name}] ident {r['ident']['est']:.3f} {r['ident']['ci']}  reversal {r['reversal']['est']:.3f} "
              f"{r['reversal']['ci']}  margins {r['margin_by_delta']}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
