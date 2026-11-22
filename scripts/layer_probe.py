"""Stage 0, inside the flat encoder: is object individuation present at some layer, even if not at the output?

On fresh clips' final, fully visible scene (the online encoder on the last tubelet), for every encoder layer:
  count : two readouts trained on frozen tokens to predict objects in the box and all objects in view
          slots - the slot-attention readout of count_readout.py (99.8% on oracle tokens)
          xfmr  - slot-free: 2-layer transformer over the tokens with a learned summary token, linear head
          Both readouts are also run on the oracle tokens as positive controls.
  attn  : per head, the attention an object token gives to the other tokens of its own object, divided by the
          share those tokens would get from uniform attention (1 = no grouping; >1 = attends within objects)

    CUDA_VISIBLE_DEVICES= python scripts/layer_probe.py RUN --weights W --data DIR --out OUT
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import webdataset as wds

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from count_readout import SlotReadout, oracle_tokens  # noqa: E402
from jepagen.shards import load_sample  # noqa: E402
from jepatrain.data import downsample  # noqa: E402
from jepatrain.reeval import load_run  # noqa: E402
from locate_surprise import MEAN, STD  # noqa: E402
from slot_probe import segments  # noqa: E402


@torch.no_grad()
def layers_and_attn(enc, x, seg):
    """x [1, 3, tb, H, W]; seg [N] (0 background). Returns per-layer tokens [L+1, N, D] (input embedding, then each
    block; the last block's output is final-normed as in Encoder.forward) and the within-object attention ratio
    [L, heads] (NaN if fewer than one object with >= 2 tokens)."""
    t = enc.embed(x).flatten(2).transpose(1, 2) + enc.pos[: enc.tokens_per_slot]
    outs, ratios = [t[0]], []
    N = t.shape[1]
    objs = [s for s in seg.unique().tolist() if s > 0 and (seg == s).sum() >= 2]
    for li, b in enumerate(enc.blocks):
        B, _, D = t.shape
        q, k, _v = b.qkv(b.n1(t)).view(B, N, 3, b.heads, D // b.heads).permute(2, 0, 3, 1, 4)
        a = torch.softmax(q @ k.transpose(-1, -2) / (D // b.heads) ** 0.5, dim=-1)[0]   # [heads, N, N]
        r = []
        for h in range(b.heads):
            vals = []
            for s in objs:
                m = seg == s
                idx = m.nonzero().flatten()
                inside = a[h][idx][:, m].sum(1) - a[h][idx, idx]              # to own object, excluding self
                base = (m.sum() - 1) / (N - 1)
                vals.append(float(inside.mean() / base))
            r.append(float(np.mean(vals)) if vals else float("nan"))
        ratios.append(r)
        t = b(t)
        outs.append(enc.norm(t)[0] if li == len(enc.blocks) - 1 else t[0])
    return torch.stack(outs), torch.tensor(ratios)


class XfmrReadout(nn.Module):
    def __init__(self, d_in, n_tok, d=64, depth=2, heads=4):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.pos = nn.Parameter(torch.randn(n_tok, d) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        z = self.inp(x) + self.pos
        z = torch.cat([self.cls.expand(len(z), -1, -1), z], 1)
        return self.head(self.tf(z)[:, 0]).squeeze(-1)


def fit(make, Xtr, ytr, Xte, yte, epochs, lr) -> float:
    torch.manual_seed(0)
    m = make()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        m.train()
        for i in torch.randperm(len(Xtr)).split(64):
            opt.zero_grad()
            F.mse_loss(m(Xtr[i].float()), ytr[i].float()).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        p = torch.cat([m(Xte[i:i + 256].float()) for i in range(0, len(Xte), 256)])
    top = int(max(ytr.max(), yte.max()))
    return float((p.round().clamp(0, top) == yte.float()).float().mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--threads", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("run with CUDA_VISIBLE_DEVICES=")
    torch.set_num_threads(a.threads)
    enc, _tgt, _pred, step = load_run(a.run, a.weights, device="cpu")
    enc.eval()
    res = json.load(open(os.path.join(a.run, "args.json")))["res"]
    L, R, O, ys = [], [], [], {"box": [], "total": []}
    for s in wds.WebDataset(sorted(glob.glob(os.path.join(a.data, "train", "possible", "*.tar"))),
                            shardshuffle=False, empty_check=False):
        ep, frames, seg, state = load_sample(s)
        last = state["frames"][-1]
        in_box = {o["id"] for o in last["objects"] if o["region"] == "box"}
        orc = oracle_tokens(seg[-1], enc.grid, in_box)
        fr = downsample(frames, res)
        x = ((torch.from_numpy(fr[-enc.tubelet:])[None].permute(0, 4, 1, 2, 3).float() / 255 - MEAN) / STD)
        toks, rat = layers_and_attn(enc, x, segments(orc[None])[0])
        L.append(toks.half()); R.append(rat); O.append(orc)
        ys["box"].append(int(ep["meta"]["final"]))
        ys["total"].append(int(last["count_per_region"]["box"] + last["count_per_region"]["hopper"]))
        if len(O) % 500 == 0:
            print(f"  {len(O)} clips", flush=True)
        if len(O) == a.n:
            break
    L, R, O = torch.stack(L), torch.stack(R), torch.stack(O)                       # [n, L+1, N, D], [n, L, h]
    ys = {k: torch.tensor(v) for k, v in ys.items()}
    tr, te = slice(0, a.n_train), slice(a.n_train, len(O))
    attn = torch.nanmean(R, 0)                                                       # [L, heads]
    print("within-object attention / uniform, per layer (max over heads, mean over heads):", flush=True)
    for li in range(attn.shape[0]):
        print(f"  layer {li + 1}: max {attn[li].max():.2f}  mean {attn[li].mean():.2f}", flush=True)
    out = {"run": a.run, "step": step, "data": a.data, "n_train": a.n_train, "n_test": len(O) - a.n_train,
           "attn_ratio": attn.tolist(), "count": {}}
    srcs = [("oracle", O)] + [(f"layer{li}", L[:, li]) for li in range(L.shape[1])]
    for tname, yy in ys.items():
        chance = float((yy[te] == torch.bincount(yy[tr]).argmax()).float().mean() * 100)
        out[f"chance_{tname}"] = chance
        print(f"target {tname}: chance {chance:.1f}%", flush=True)
        for name, X in srcs:
            d, n = X.shape[-1], X.shape[1]
            r = {"slots": fit(lambda: SlotReadout(d, n), X[tr], yy[tr], X[te], yy[te], a.epochs, 4e-4),
                 "xfmr": fit(lambda: XfmrReadout(d, n), X[tr], yy[tr], X[te], yy[te], a.epochs, 3e-4)}
            out["count"][f"{tname}/{name}"] = r
            print(f"  {name:<8} slots {r['slots']:5.1f}%  xfmr {r['xfmr']:5.1f}%", flush=True)
    os.makedirs(a.out, exist_ok=True)
    name = f"{os.path.basename(a.run.rstrip('/'))}_{step:06d}"
    json.dump(out, open(os.path.join(a.out, f"layers_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
