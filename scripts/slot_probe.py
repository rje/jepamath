"""Can object slots be found in a trained Track B encoder's features? (stage 0, slot branches)

Slot attention (K slots) over frozen token features of the final, fully visible scene (the cache written by
count_readout.py --features), trained to reconstruct those features through a spatial-broadcast decoder whose
per-slot outputs are mixed by per-position softmax weights (DINOSAUR, Seitzer et al. 2023):
  free : reconstruction only (mask-free branch)
  mask : reconstruction + cross-entropy pulling each token's slot assignment toward its Hungarian-matched object
         segment (mask-supervised branch; segments from the oracle tokens, background is its own segment)
Scored on held-out clips:
  fg_ari : adjusted Rand index between slot assignment (argmax decoder weight) and objects, on object tokens
  count  : the slot vectors (frozen) feed a per-slot presence head summing to the count (box / all in view),
           trained on the training clips; the same readout reached 99.8% on oracle tokens

    CUDA_VISIBLE_DEVICES= python scripts/slot_probe.py --features FEATS.pt --source enc --out OUT
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
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(__file__))
from count_readout import ID_CODES  # noqa: E402


def segments(oracle: torch.Tensor) -> torch.Tensor:
    """oracle [B, N, 2 + ID_DIM] -> [B, N] long: 0 background, else 1 + object id (from the fixed id codes)."""
    codes = oracle[..., 2:]
    d = torch.cdist(codes.reshape(-1, codes.shape[-1]), ID_CODES)               # [B*N, N_IDS]
    ids = d.argmin(-1).view(oracle.shape[:2])
    return torch.where(oracle[..., 0] > 0.5, ids + 1, torch.zeros_like(ids))


class SlotAE(nn.Module):
    def __init__(self, d_in, n_tok, k=8, d=128, iters=3):
        super().__init__()
        self.k, self.d, self.iters = k, d, iters
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.ReLU(), nn.Linear(d, d))
        self.pos_in = nn.Parameter(torch.randn(n_tok, d) * 0.02)
        self.mu = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        self.logsig = nn.Parameter(torch.zeros(1, 1, d))
        self.q, self.kp, self.vp = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.gru = nn.GRUCell(d, d)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.ns, self.nk = nn.LayerNorm(d), nn.LayerNorm(d)
        self.pos_dec = nn.Parameter(torch.randn(n_tok, d) * 0.02)
        self.dec = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, 2 * d), nn.ReLU(),
                                 nn.Linear(2 * d, d_in + 1))

    def slots(self, x):
        B = x.shape[0]
        x = self.nk(self.inp(x) + self.pos_in)
        k, v = self.kp(x), self.vp(x)
        s = self.mu + self.logsig.exp() * torch.randn(B, self.k, self.d, device=x.device)
        for _ in range(self.iters):
            q = self.q(self.ns(s))
            a = torch.softmax(torch.einsum("bkd,bnd->bkn", q, k) / math.sqrt(self.d), dim=1)
            a = a / a.sum(-1, keepdim=True).clamp_min(1e-8)
            s = self.gru(torch.einsum("bkn,bnd->bkd", a, v).reshape(-1, self.d), s.reshape(-1, self.d))
            s = s.view(B, self.k, self.d)
            s = s + self.mlp(s)
        return s

    def forward(self, x):
        s = self.slots(x)                                                     # [B, K, d]
        out = self.dec(s[:, :, None] + self.pos_dec[None, None])               # [B, K, N, d_in + 1]
        w = torch.softmax(out[..., -1], dim=1)                                 # per position, over slots
        recon = (w[..., None] * out[..., :-1]).sum(1)                          # [B, N, d_in]
        return s, w, recon


def mask_loss(w, seg):
    """w [B, K, N] decoder weights; seg [B, N]. Hungarian-match segments to slots by overlap, then cross-entropy of
    each token's slot distribution toward its segment's slot."""
    B, K, N = w.shape
    logw = torch.log(w.clamp_min(1e-8))
    target = torch.zeros(B, N, dtype=torch.long)
    with torch.no_grad():
        for b in range(B):
            labs = seg[b].unique()
            if len(labs) > K:
                raise ValueError(f"{len(labs)} segments > {K} slots")
            ov = torch.stack([w[b][:, seg[b] == l].sum(1) for l in labs])     # [n_seg, K]
            r, c = linear_sum_assignment(-ov.numpy())
            for ri, ci in zip(r, c):
                target[b][seg[b] == labs[ri]] = int(ci)
    return F.nll_loss(logw.permute(0, 2, 1).reshape(-1, K), target.reshape(-1))


def fg_ari(w, seg) -> float:
    assign = w.argmax(1)
    scores = []
    for b in range(len(seg)):
        m = seg[b] > 0
        if m.sum() > 1 and len(seg[b][m].unique()) > 1:
            scores.append(adjusted_rand_score(seg[b][m].numpy(), assign[b][m].numpy()))
    return float(np.mean(scores))


class Presence(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.h = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, s):
        return torch.sigmoid(self.h(s)).squeeze(-1).sum(-1)


def count_acc(Str, ytr, Ste, yte, epochs=150) -> float:
    torch.manual_seed(0)
    m = Presence(Str.shape[-1])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        for i in torch.randperm(len(Str)).split(128):
            opt.zero_grad()
            F.mse_loss(m(Str[i]), ytr[i].float()).backward()
            opt.step()
    with torch.no_grad():
        p = m(Ste).round().clamp(0, int(max(ytr.max(), yte.max())))
    return float((p == yte.float()).float().mean() * 100)


@torch.no_grad()
def encode(model, X, bs=256):
    S, W = [], []
    for i in range(0, len(X), bs):
        s, w, _ = model(X[i:i + bs])
        S.append(s); W.append(w)
    return torch.cat(S), torch.cat(W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="cache from count_readout.py --features")
    ap.add_argument("--source", default="enc", choices=("enc", "tgt", "pixels", "oracle"))
    ap.add_argument("--mode", default="free", choices=("free", "mask"))
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--mask-weight", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("run with CUDA_VISIBLE_DEVICES=")
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    c = torch.load(a.features)
    X, seg, ys = c["sets"][a.source].float(), segments(c["sets"]["oracle"]), c["ys"]
    del c
    tr, te = slice(0, a.n_train), slice(a.n_train, len(X))
    model = SlotAE(X.shape[-1], X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4)
    for ep in range(a.epochs):
        model.train()
        tot = {"rec": 0.0, "mask": 0.0}
        for i in torch.randperm(a.n_train).split(64):
            _, w, recon = model(X[i])
            rec = F.mse_loss(recon, X[i])
            loss = rec
            if a.mode == "mask":
                ml = mask_loss(w, seg[i])
                loss = loss + a.mask_weight * ml
                tot["mask"] += float(ml)
            opt.zero_grad(); loss.backward(); opt.step()
            tot["rec"] += float(rec)
        if ep % 5 == 4 or ep == a.epochs - 1:
            model.eval()
            _, wte = encode(model, X[te][:500])
            print(f"  epoch {ep + 1}: rec {tot['rec']:.2f} mask {tot['mask']:.2f}  fg_ari(500 test) "
                  f"{fg_ari(wte, seg[te][:500]):.3f}", flush=True)
    model.eval()
    S, W = encode(model, X)
    out = {"features": a.features, "source": a.source, "mode": a.mode, "epochs": a.epochs,
           "fg_ari_test": fg_ari(W[te], seg[te]),
           "count_box": count_acc(S[tr], ys["box"][tr], S[te], ys["box"][te]),
           "count_total": count_acc(S[tr], ys["total"][tr], S[te], ys["total"][te])}
    print(json.dumps(out), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, f"slots_{a.source}_{a.mode}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
