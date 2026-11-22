"""Stage-0/5 count readout: can the final object count be read from a Track B model's features?

On the possible clips of a held-out set, take the last time slot (the box fully visible, all objects at rest):
  true : target-encoder features of the real clip (what the model sees)        -> stage 0
  pred : predictor output from the pre-reveal context (what the model expects)  -> stage 5
and fit three probes on the first half, scoring the second half:
  pixels : the same probes on the final frame's raw 8x8 patches, a positive control for the probes themselves
  mean   : logistic regression on token-averaged features (the training-time probe; washes small objects out)
  sum    : count = sum over tokens of softplus(w.h + b) (an "object-ness" density; behaves like total area)
  attn   : 4 learned queries attend over tokens, then a linear 4-way classifier (can pick out individuals)
On area-controlled data (v4b) area does not predict count, so `sum` should fail and only individuated
features let `attn` succeed. Chance = the majority class of the test half.

    CUDA_VISIBLE_DEVICES= python scripts/count_probe.py RUN --weights W --heldout H --out OUT
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from jepatrain.reeval import load_run  # noqa: E402
from locate_surprise import MEAN, STD, timeline  # noqa: E402


@torch.no_grad()
def features(enc, tgt, pred, t: dict) -> tuple[torch.Tensor, torch.Tensor]:
    tb, G = enc.tubelet, enc.grid * enc.grid
    fr = t["possible"]
    T = fr.shape[0]
    ctx, _ = timeline(t["pos_episode"], T, tb)
    x = ((torch.from_numpy(fr)[None].permute(0, 4, 1, 2, 3).float() / 255 - MEAN) / STD)
    h = tgt(x).float()
    h = F.layer_norm(h, (h.size(-1),))[0]                                       # [N, D]
    n_ctx, N = ctx // tb * G, T // tb * G
    ar = lambda a, b: torch.arange(a, b).unsqueeze(0)
    p = pred(enc(x[:, :, :ctx]), ar(0, n_ctx), ar(n_ctx, N)).float()[0]          # [N - n_ctx, D]
    # positive control: the final frame's raw 8x8 RGB patches as tokens (the count is plainly visible there)
    pz = enc.patch
    f = torch.from_numpy(fr[-1]).float() / 255                                   # [H, W, 3]
    g = f.shape[0] // pz
    pix = f.view(g, pz, g, pz, 3).permute(0, 2, 1, 3, 4).reshape(g * g, -1)     # [G, pz*pz*3]
    return h[-G:], p[-G:], pix                                                   # last slot, [G, D] each


class Attn(nn.Module):
    def __init__(self, d, q=4, k=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(q, d) / math.sqrt(d))
        self.kp, self.vp = nn.Linear(d, d), nn.Linear(d, 64)
        self.out = nn.Linear(q * 64, k)

    def forward(self, x):                                                        # x [B, G, D]
        a = torch.softmax(self.kp(x) @ self.q.T / math.sqrt(x.shape[-1]), dim=1)  # [B, G, q]
        v = torch.einsum("bgq,bgd->bqd", a, self.vp(x))
        return self.out(v.flatten(1))


class Sum(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)
        nn.init.zeros_(self.w.weight)
        nn.init.constant_(self.w.bias, -6.0)   # start near a count of 0 over ~256 tokens, not ~177

    def forward(self, x):
        return F.softplus(self.w(x)).sum((1, 2))


def fit(model, X, y, loss_fn, epochs=300, lr=3e-3, wd=1e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        for i in torch.randperm(len(X)).split(128):
            opt.zero_grad()
            loss_fn(model(X[i]), y[i]).backward()
            opt.step()
    return model


def probes(Xtr, ytr, Xte, yte) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr.mean(1).numpy())
    lr = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr.mean(1).numpy()), ytr.numpy())
    mean_acc = float((lr.predict(sc.transform(Xte.mean(1).numpy())) == yte.numpy()).mean() * 100)
    s = fit(Sum(Xtr.shape[-1]), Xtr, ytr.float(), F.mse_loss)
    with torch.no_grad():
        sum_acc = float((s(Xte).round().clamp(0, 3) == yte.float()).float().mean() * 100)
    m = fit(Attn(Xtr.shape[-1]), Xtr, ytr, F.cross_entropy)
    with torch.no_grad():
        attn_acc = float((m(Xte).argmax(1) == yte).float().mean() * 100)
    return {"mean": mean_acc, "sum": sum_acc, "attn": attn_acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("run with CUDA_VISIBLE_DEVICES=")
    torch.set_num_threads(a.threads)
    enc, tgt, pred, step = load_run(a.run, a.weights, device="cpu")
    for m in (enc, tgt, pred):
        m.eval()
    triples = torch.load(a.heldout, weights_only=False)[: a.n]
    H, P, X, y = [], [], [], []
    for i, t in enumerate(triples):
        h, p, pix = features(enc, tgt, pred, t)
        H.append(h); P.append(p); X.append(pix); y.append(int(t["meta"]["final"]))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(triples)}", flush=True)
    H, P, X, y = torch.stack(H), torch.stack(P), torch.stack(X), torch.tensor(y)
    if y.max() > 3 or y.min() < 0:
        raise ValueError(f"counts outside 0-3: {y.unique().tolist()} (the probes assume v4's range)")
    h = len(y) // 2
    chance = float((y[h:] == torch.bincount(y[:h]).argmax()).float().mean() * 100)
    res = {"run": a.run, "step": step, "heldout": a.heldout, "n_test": len(y) - h, "chance": chance,
           "pixels": probes(X[:h], y[:h], X[h:], y[h:]),
           "true": probes(H[:h], y[:h], H[h:], y[h:]), "pred": probes(P[:h], y[:h], P[h:], y[h:]),
           "counts": torch.bincount(y).tolist()}
    os.makedirs(a.out, exist_ok=True)
    name = f"{os.path.basename(a.run.rstrip('/'))}_{step:06d}_{os.path.basename(a.heldout).split('_')[0]}"
    json.dump(res, open(os.path.join(a.out, f"count_{name}.json"), "w"), indent=1)
    print(f"{name}  chance {chance:.1f}%  counts {res['counts']}")
    for k in ("pixels", "true", "pred"):
        print(f"  {k:<5} " + "  ".join(f"{p} {v:5.1f}%" for p, v in res[k].items()))


if __name__ == "__main__":
    main()
