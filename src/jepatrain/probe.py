"""Attentive count probes on a Track B checkpoint (stronger readout than the mean-pooled linear probes).

Features are the token grids of one time slot (grid**2 tokens):
  start      : encoder on the first tubelet only            -> label n0
  post_true  : target encoder, first slot after the reveal  -> label final
  post_pred  : predictor's output for that slot, from the pre-lowering context -> read with the post_true probe
Probes are trained on possible clips from the training shards (self-supervised training never saw labels) and
tested on the held-out triples' possible clips.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import webdataset as wds

from jepaeval.guard import ComputeGuard
from jepaeval.twins import _screen_times

from .data import downsample
from .evaluate import FPS, TrainedSurprise
from .model import Encoder, Predictor


def train_clips(data: str, exclude: set[str], n: int, split: str = "train"):
    shards = sorted(glob.glob(os.path.join(data, split, "possible", "*.tar")))
    out = []
    for s in wds.WebDataset(shards, shardshuffle=False, empty_check=False):
        if s["__key__"] in exclude:
            continue
        ep = json.loads(s["episode.json"])
        out.append((downsample(np.load(io.BytesIO(s["clip.npz"]))["frames"]), ep))
        if len(out) == n:
            return out
    raise RuntimeError(f"only {len(out)} training clips found, wanted {n}")


@torch.no_grad()
def slot_tokens(sj: TrainedSurprise, frames, ep) -> dict:
    x = sj.preprocess(frames).unsqueeze(0)
    tb, G, T = sj.tubelet, sj.grid * sj.grid, x.shape[2]
    t0, t1 = _screen_times(ep)
    ctx = int(t0 * FPS) // tb * tb
    f_end = int(np.ceil(t1 * FPS))
    first = next(j for j in range((T - ctx) // tb) if ctx + tb * (j + 1) - 1 >= f_end)
    n_ctx, N = ctx // tb * G, T // tb * G
    ar = lambda a, b: torch.arange(a, b, device=x.device).unsqueeze(0)
    with sj.guard, torch.autocast("cuda", dtype=torch.bfloat16):
        start = sj.enc(x[:, :, :tb])[0]
        h = sj.tgt(x).float()
        h = F.layer_norm(h, (h.size(-1),))[0]
        p = sj.pred(sj.enc(x[:, :, :ctx]), ar(0, n_ctx), ar(n_ctx, N))[0]
    s0 = n_ctx + first * G
    return {"start": start.half().cpu(), "post_true": h[s0:s0 + G].half().cpu(),
            "post_pred": p[first * G:(first + 1) * G].half().cpu()}


class AttentiveProbe(nn.Module):
    def __init__(self, dim, n_classes, heads=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_classes))

    def forward(self, t):
        t = self.norm(t)
        o, _ = self.attn(self.q.expand(t.shape[0], -1, -1), t, t)
        return self.head(o[:, 0])


def fit_probe(Xtr, ytr, n_classes, epochs=40, device="cuda", seed=0):
    torch.manual_seed(seed)
    m = AttentiveProbe(Xtr.shape[-1], n_classes).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
    n = len(Xtr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * ((n + 63) // 64))
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 64):
            b = perm[i:i + 64]
            loss = F.cross_entropy(m(Xtr[b].to(device).float()), ytr[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m.eval()


@torch.no_grad()
def acc(m, X, y, device="cuda"):
    pr = torch.cat([m(X[i:i + 256].to(device).float()).argmax(1).cpu() for i in range(0, len(X), 256)])
    return float((pr == y).float().mean() * 100)


def main():
    ap = argparse.ArgumentParser(prog="jepatrain.probe")
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", required=True, help="weights_XXXXXX.pt in the run dir")
    ap.add_argument("--data", default="/mnt/data/jepamath/v2")
    ap.add_argument("--heldout", default="/mnt/data/jepamath/heldout/v2_triples2000_128.pt")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--duty", type=float, default=0.5)
    a = ap.parse_args()
    args = json.load(open(os.path.join(a.run, "args.json")))
    enc = Encoder(dim=args["dim"], depth=args["depth"], heads=args["dim"] // 64).cuda()
    pred = Predictor(enc, dim=args["pred_dim"], depth=args["pred_depth"], heads=args["pred_dim"] // 32).cuda()
    tgt = Encoder(dim=args["dim"], depth=args["depth"], heads=args["dim"] // 64).cuda()
    w = torch.load(os.path.join(a.run, a.weights), map_location="cuda")
    enc.load_state_dict(w["enc"]); pred.load_state_dict(w["pred"]); tgt.load_state_dict(w["tgt"])
    for m in (enc, pred, tgt):
        m.eval()
    sj = TrainedSurprise(enc, tgt, pred, ComputeGuard(duty=a.duty))
    triples = torch.load(a.heldout, weights_only=False)
    exclude = {t["seed_id"] for t in triples}
    tr = train_clips(a.data, exclude, a.n_train)
    te = [(t["possible"], t["pos_episode"]) for t in triples]

    def feats(clips):
        F_ = [slot_tokens(sj, f, ep) for f, ep in clips]
        lab = lambda k: torch.tensor([ep["meta"][k] for _, ep in clips])
        return {k: torch.stack([f[k] for f in F_]) for k in F_[0]}, lab("n0"), lab("final")

    Ftr, n0tr, fintr = feats(tr)
    Fte, n0te, finte = feats(te)
    nc = int(max(fintr.max(), finte.max(), n0tr.max(), n0te.max())) + 1
    p_n0 = fit_probe(Ftr["start"], n0tr, nc)
    p_fin = fit_probe(Ftr["post_true"], fintr, nc)
    chg = n0te != finte
    res = {"weights": a.weights, "n_train": len(tr), "n_test": len(te),
           "n0@start": acc(p_n0, Fte["start"], n0te),
           "final@post_true": acc(p_fin, Fte["post_true"], finte),
           "final@post_pred": acc(p_fin, Fte["post_pred"], finte),
           "n0@post_pred": acc(p_fin, Fte["post_pred"], n0te),
           "final@post_pred|changed": acc(p_fin, Fte["post_pred"][chg], finte[chg]),
           "n0@post_pred|changed": acc(p_fin, Fte["post_pred"][chg], n0te[chg]),
           "chance_final": float((finte == torch.bincount(fintr).argmax()).float().mean() * 100)}
    print(json.dumps(res))
    with open(os.path.join(a.run, "attentive_probes.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
