"""Stage 0: does a Track B encoder represent the objects in a visible scene as countable units?

Fresh clips (rendered with segmentation, never trained on); the last frame shows every object at rest. For each
clip, token sets from four sources describe that final scene:
  tgt    : target-encoder features of the last time slot of the clip (what the model sees, in context)
  enc    : online encoder on the last tubelet alone (the static final scene)
  oracle : POSITIVE CONTROL from the segmentation masks: per token, "is an object" plus a fixed random code for
           which object, i.e. perfect individuation. A readout that cannot count these is broken.
  pixels : raw 8x8 RGB patches
and readouts are trained on frozen tokens to predict two counts: objects in the box (0-3; hopper objects are also
in view and must be ignored, so the tray matters) and all objects in view (box + hopper; individuation alone):
  mean   : logistic regression on token-averaged features (the training-time probe)
  slots  : slot attention (6 slots, 3 iterations) with a per-slot presence head; count = sum of presences
On area-controlled data (v4b) total object area does not give the count, so success needs individuation.

    CUDA_VISIBLE_DEVICES= python scripts/count_readout.py RUN --weights W --data DIR --out OUT
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import webdataset as wds

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from jepagen.shards import load_sample  # noqa: E402
from jepatrain.data import downsample  # noqa: E402
from jepatrain.reeval import load_run  # noqa: E402
from locate_surprise import MEAN, STD  # noqa: E402

N_IDS, ID_DIM = 32, 16
ID_CODES = torch.randn(N_IDS, ID_DIM, generator=torch.Generator().manual_seed(0))


def oracle_tokens(seg_last: np.ndarray, grid: int, in_box: set[int]) -> torch.Tensor:
    """seg [H, W] (0 = background, 1..250 = object id + 1) -> [grid*grid, 2 + ID_DIM]: per patch, whether an
    object covers enough of it, whether that object is in the box (the trays are not in the segmentation, so the
    simulator's final regions supply it), and the fixed code of the object covering the most pixels."""
    H = seg_last.shape[0]
    p = H // grid
    blocks = seg_last.reshape(grid, p, grid, p).transpose(0, 2, 1, 3).reshape(grid * grid, p * p)
    out = torch.zeros(grid * grid, 2 + ID_DIM)
    for i, b in enumerate(blocks):
        ids = b[(b >= 1) & (b <= 250)]
        if len(ids) * 2 >= len(b) // 4:          # object covers at least an eighth of the patch
            oid = int(np.bincount(ids).argmax())
            if oid >= N_IDS:
                raise ValueError(f"object id {oid} >= {N_IDS}")
            out[i, 0] = 1.0
            out[i, 1] = float(oid - 1 in in_box)
            out[i, 2:] = ID_CODES[oid]
    return out


@torch.no_grad()
def token_sets(enc, tgt, frames: np.ndarray, seg: np.ndarray, res: int, in_box: set[int]) -> dict[str, torch.Tensor]:
    tb, G = enc.tubelet, enc.grid * enc.grid
    fr = downsample(frames, res)
    x = ((torch.from_numpy(fr)[None].permute(0, 4, 1, 2, 3).float() / 255 - MEAN) / STD)
    h = tgt(x).float()
    h = F.layer_norm(h, (h.size(-1),))[0, -G:]
    e = enc(x[:, :, -tb:]).float()[0]                                       # the last tubelet alone: [G, D]
    pz = res // enc.grid
    f = torch.from_numpy(fr[-1]).float() / 255
    pix = f.view(enc.grid, pz, enc.grid, pz, 3).permute(0, 2, 1, 3, 4).reshape(G, -1)
    return {"tgt": h, "enc": e, "oracle": oracle_tokens(seg[-1], enc.grid, in_box), "pixels": pix}


class SlotReadout(nn.Module):
    def __init__(self, d_in, n_tok, k=8, d=64, iters=3):
        super().__init__()
        self.k, self.d, self.iters = k, d, iters
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.pos = nn.Parameter(torch.randn(n_tok, d) * 0.02)
        self.mu = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        self.logsig = nn.Parameter(torch.zeros(1, 1, d))
        self.q, self.kp, self.vp = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.gru = nn.GRUCell(d, d)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.ns, self.nk = nn.LayerNorm(d), nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x):                                                   # x [B, N, d_in] -> count [B]
        B = x.shape[0]
        x = self.nk(self.inp(x) + self.pos)
        k, v = self.kp(x), self.vp(x)
        s = self.mu + self.logsig.exp() * torch.randn(B, self.k, self.d, device=x.device)
        for _ in range(self.iters):
            q = self.q(self.ns(s))
            a = torch.softmax(torch.einsum("bkd,bnd->bkn", q, k) / math.sqrt(self.d), dim=1)  # slots compete
            a = a / a.sum(-1, keepdim=True).clamp_min(1e-8)
            u = torch.einsum("bkn,bnd->bkd", a, v)
            s = self.gru(u.reshape(-1, self.d), s.reshape(-1, self.d)).view(B, self.k, self.d)
            s = s + self.mlp(s)
        return torch.sigmoid(self.head(s)).squeeze(-1).sum(-1)


def fit_slots(Xtr, ytr, Xte, yte, epochs, seed=0) -> float:
    top = int(max(ytr.max(), yte.max()))
    torch.manual_seed(seed)
    m = SlotReadout(Xtr.shape[-1], Xtr.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=4e-4, weight_decay=1e-4)
    for ep in range(epochs):
        m.train()
        for i in torch.randperm(len(Xtr)).split(64):
            opt.zero_grad()
            F.mse_loss(m(Xtr[i]), ytr[i].float()).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        pred = torch.cat([m(Xte[i:i + 256]) for i in range(0, len(Xte), 256)])
    return float((pred.round().clamp(0, top) == yte.float()).float().mean() * 100)


def fit_mean(Xtr, ytr, Xte, yte) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr.mean(1).numpy())
    lr = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr.mean(1).numpy()), ytr.numpy())
    return float((lr.predict(sc.transform(Xte.mean(1).numpy())) == yte.numpy()).mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="generated clips with segmentation (not trained on)")
    ap.add_argument("--n", type=int, default=7000)
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--threads", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--features", help="token-set cache (.pt): written if absent, read if present")
    a = ap.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("run with CUDA_VISIBLE_DEVICES=")
    torch.set_num_threads(a.threads)
    enc, tgt, pred, step = load_run(a.run, a.weights, device="cpu")
    for m in (enc, tgt):
        m.eval()
    res = json.load(open(os.path.join(a.run, "args.json")))["res"]
    sets, ys = {}, {"box": [], "total": []}
    cached = a.features and os.path.exists(a.features)
    if cached:
        c = torch.load(a.features)
        if (c["run"], c["step"], c["data"]) != (a.run, step, a.data):
            raise ValueError(f"{a.features} holds {c['run']} step {c['step']} on {c['data']}, not this run")
        sets, ys = c["sets"], c["ys"]
    for s in ([] if cached else wds.WebDataset(sorted(glob.glob(os.path.join(a.data, "train", "possible", "*.tar"))),
                            shardshuffle=False, empty_check=False)):
        ep, frames, seg, state = load_sample(s)
        last = state["frames"][-1]
        if ep["meta"]["final"] != last["count_per_region"]["box"]:
            raise ValueError(f"{ep['episode_id']}: meta final {ep['meta']['final']} != simulated box count")
        in_box = {o["id"] for o in last["objects"] if o["region"] == "box"}
        for k, v in token_sets(enc, tgt, frames, seg, res, in_box).items():
            sets.setdefault(k, []).append(v)
        ys["box"].append(int(ep["meta"]["final"]))
        ys["total"].append(int(last["count_per_region"]["box"] + last["count_per_region"]["hopper"]))
        if len(ys["box"]) % 1000 == 0:
            print(f"  {len(ys['box'])} clips", flush=True)
        if len(ys["box"]) == a.n:
            break
    if not cached:
        ys = {k: torch.tensor(v) for k, v in ys.items()}
        sets = {k: torch.stack(v) for k, v in sets.items()}
        if a.features:
            torch.save({"run": a.run, "step": step, "data": a.data, "sets": sets, "ys": ys}, a.features)
    y = ys["box"]
    if len(y) <= a.n_train:
        raise ValueError(f"only {len(y)} clips for {a.n_train} training")
    tr, te = slice(0, a.n_train), slice(a.n_train, len(y))
    out = {"run": a.run, "step": step, "data": a.data, "n_train": a.n_train, "n_test": len(y) - a.n_train, "acc": {}}
    for tname, yy in ys.items():
        chance = float((yy[te] == torch.bincount(yy[tr]).argmax()).float().mean() * 100)
        out[f"chance_{tname}"], out[f"counts_{tname}"] = chance, torch.bincount(yy).tolist()
        print(f"target {tname}: chance {chance:.1f}%  counts {out[f'counts_{tname}']}", flush=True)
        for k in ("oracle", "tgt", "enc", "pixels"):
            X = sets[k]
            r = {"mean": fit_mean(X[tr], yy[tr], X[te], yy[te]), "slots": fit_slots(X[tr], yy[tr], X[te], yy[te], a.epochs)}
            out["acc"][f"{tname}/{k}"] = r
            print(f"  {k:<7} mean {r['mean']:5.1f}%   slots {r['slots']:5.1f}%", flush=True)
    os.makedirs(a.out, exist_ok=True)
    name = f"{os.path.basename(a.run.rstrip('/'))}_{step:06d}"
    json.dump(out, open(os.path.join(a.out, f"countread_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
