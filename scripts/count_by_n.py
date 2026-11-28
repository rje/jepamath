"""Per-count breakdown of the stage-0 count readout (transformer readout on one encoder layer).
For each true count n: test accuracy, mean prediction, mean abs error. Number-sense signatures to look for:
accuracy falling with n (Weber-like), small n exact (subitizing), errors scaling with n."""
import argparse, glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layer_probe as lp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--weights", required=True)
ap.add_argument("--data", default="/mnt/data/jepamath/v4b_probe")
ap.add_argument("--layer", type=int, default=8); ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--n-train", type=int, default=1400); ap.add_argument("--out", required=True)
ap.add_argument("--train-max", type=int, nargs=2, metavar=("BOX", "TOTAL"), default=None,
                help="extrapolation test: fit the readout only on training clips with count <= BOX / TOTAL")
ap.add_argument("--readout", choices=("xfmr", "linear-mean"), nargs="+", default=["xfmr"],
                help="xfmr: the transformer readout of layer_probe; linear-mean: ridge regression on mean-pooled tokens (a linear "
                     "map can extrapolate past the counts it was fit on; the transformer tends to clamp). Sum pooling "
                     "would be redundant: every clip has the same token count, and features are standardised")
ap.add_argument("--ridge", type=float, default=1.0)
ap.add_argument("--test-data", default=None,
                help="generalization test: fit the readout on --data, test on the first n - n_train clips of this set")
ap.add_argument("--moved-thresh", type=float, default=0.02,
                help="motion shortcut test: a box object 'moved' if its max displacement over the clip exceeds this (m)")
ap.add_argument("--encoder", choices=("context", "target"), default="context",
                help="read the context encoder or the EMA target encoder (same architecture)")
ap.add_argument("--threads", type=int, default=40, help="CPU threads for feature extraction and readout fitting")
a = ap.parse_args()
if torch.cuda.is_available():
    raise SystemExit("run with CUDA_VISIBLE_DEVICES=")
torch.set_num_threads(a.threads)
enc, tgt, _p, step = lp.load_run(a.run, a.weights, device="cpu")
if a.encoder == "target":
    enc = tgt
enc.eval()
res = json.load(open(os.path.join(a.run, "args.json")))["res"]
def _max_disp(frames, oid) -> float:
    """Largest distance an object gets from where it was first seen, over the clip."""
    ps = np.array([o["pos"] for f in frames for o in f["objects"] if o["id"] == oid])
    return float(np.linalg.norm(ps - ps[0], axis=1).max())


def extract(data: str, n: int):
    X, ys, modes, moved = [], {"box": [], "total": []}, [], []
    for s in wds.WebDataset(sorted(glob.glob(os.path.join(data, "train", "possible", "*.tar"))),
                            shardshuffle=False, empty_check=False):
        ep, frames, seg, state = lp.load_sample(s)
        last = state["frames"][-1]
        in_box = {o["id"] for o in last["objects"] if o["region"] == "box"}
        orc = lp.oracle_tokens(seg[-1], enc.grid, in_box)
        fr = lp.downsample(frames, res)
        x = ((torch.from_numpy(fr[-enc.tubelet:])[None].permute(0, 4, 1, 2, 3).float() / 255 - lp.MEAN) / lp.STD)
        toks, _ = lp.layers_and_attn(enc, x, lp.segments(orc[None])[0])
        X.append(toks[a.layer].half())
        ys["box"].append(int(ep["meta"]["final"]))
        modes.append(ep["meta"]["control_mode"])
        moved.append(sum(_max_disp(state["frames"], o["id"]) > a.moved_thresh
                         for o in last["objects"] if o["region"] == "box"))
        ys["total"].append(int(last["count_per_region"]["box"] + last["count_per_region"]["hopper"]))
        if len(X) == n:
            break
    if len(X) < n:
        raise ValueError(f"{data}: only {len(X)} clips, need {n}")
    return X, ys, modes, moved


if a.test_data is None:
    X, ys, modes, moved = extract(a.data, a.n)
else:                       # fit on a.data, test on a.test_data (generalization: same readout, new clips)
    X, ys, modes, moved = extract(a.data, a.n_train)
    Xt, yt, mt, mvt = extract(a.test_data, a.n - a.n_train)
    X += Xt; modes += mt; moved += mvt
    ys = {k: ys[k] + yt[k] for k in ys}
X = torch.stack(X); tr, te = slice(0, a.n_train), slice(a.n_train, len(X))
out = {"run": a.run, "step": step, "layer": a.layer, "train_max": a.train_max, "encoder": a.encoder, "data": a.data, "test_data": a.test_data}
for readout in a.readout:
    for tname, yy in ys.items():
        yy = torch.tensor(yy)
        keep = torch.arange(a.n_train)
        if a.train_max is not None:
            keep = keep[yy[tr] <= a.train_max[0 if tname == "box" else 1]]
        Xf, yf = X[tr][keep], yy[tr][keep]
        if readout == "xfmr":
            torch.manual_seed(0)
            m = lp.XfmrReadout(X.shape[-1], X.shape[1])
            opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
            for _ in range(40):
                m.train()
                for i in torch.randperm(len(yf)).split(64):
                    opt.zero_grad(); F.mse_loss(m(Xf[i].float()), yf[i].float()).backward(); opt.step()
            m.eval()
            with torch.no_grad():
                p = torch.cat([m(X[te][i:i + 256].float()) for i in range(0, len(X[te]), 256)])
        else:
            Ptr, Pte = Xf.float().mean(1), X[te].float().mean(1)
            mu, sd = Ptr.mean(0), Ptr.std(0) + 1e-6                  # standardise on the fit set only
            Ptr, Pte = (Ptr - mu) / sd, (Pte - mu) / sd
            A = torch.cat([Ptr, torch.ones(len(Ptr), 1)], 1).double()
            W = torch.linalg.solve(A.T @ A + a.ridge * torch.eye(A.shape[1], dtype=A.dtype), A.T @ yf.double()[:, None])
            p = (torch.cat([Pte, torch.ones(len(Pte), 1)], 1).double() @ W)[:, 0].float()
        top = int(yy.max()); pr = p.round().clamp(0, top); t = yy[te]
        rows = []
        print(f"target {tname} ({readout}, layer {a.layer}, fit on {len(yf)} clips, overall {float((pr == t).float().mean()) * 100:.1f}%)")
        print("   n   count   acc%   mean_pred   MAE")
        for n in range(top + 1):
            k = t == n
            if k.sum() == 0:
                continue
            r = {"n": n, "count": int(k.sum()), "acc": float((pr[k] == n).float().mean() * 100),
                 "mean_pred": float(p[k].mean()), "mae": float((p[k] - n).abs().mean())}
            rows.append(r)
            print(f"  {n:2d}   {r['count']:5d}  {r['acc']:5.1f}   {r['mean_pred']:8.2f}  {r['mae']:5.2f}")
        out[f"{readout}/{tname}"] = rows
        out[f"{readout}/{tname}/pred"] = [float(v) for v in p]     # raw test predictions, for bootstrap CIs
        out[f"{readout}/{tname}/pred_int"] = [int(v) for v in pr]  # rounded and clamped, as scored above
        # shortcut control: in area_matched clips the box objects shrink as the final box count grows (total area
        # ~constant within the radius range), so an area code cannot tell box counts apart there
        mte = modes[a.n_train:]
        for mode in sorted(set(mte)):
            k = torch.tensor([m_ == mode for m_ in mte])
            per = {n: (round(float((pr[k & (t == n)] == n).float().mean() * 100), 1), round(float(p[k & (t == n)].mean()), 2))
                   for n in range(top + 1) if (k & (t == n)).any()}
            print(f"   {mode:13s} acc {float((pr[k] == t[k]).float().mean()) * 100:5.1f}%   by n (acc%, mean_pred): {per}")
            out[f"{readout}/{tname}/{mode}"] = {"acc": float((pr[k] == t[k]).float().mean() * 100), "by_n": per}
        if tname == "box":
            # motion shortcut test: does the prediction track box objects that moved, or all box objects?
            mv = np.array(moved[a.n_train:]); st_ = t.numpy() - mv; pn = p.numpy()
            cells = {f"{n}box/{m}moved": (int(((t.numpy() == n) & (mv == m)).sum()),
                                           round(float((pr.numpy()[(t.numpy() == n) & (mv == m)] == n).mean() * 100), 1),
                                           round(float(pn[(t.numpy() == n) & (mv == m)].mean()), 2))
                     for n in range(top + 1) for m in range(n + 1) if ((t.numpy() == n) & (mv == m)).sum() >= 5}
            A = np.c_[mv, st_, np.ones(len(mv))]
            coef = np.linalg.lstsq(A, pn, rcond=None)[0]
            print(f"   by (box, moved): (clips, acc%, mean_pred) {cells}")
            print(f"   pred ~ {coef[0]:.2f} * moved + {coef[1]:.2f} * still + {coef[2]:.2f}   (a true count gives 1.00 and 1.00)")
            out[f"{readout}/box/moved_test"] = {"cells": cells, "coef_moved": float(coef[0]), "coef_still": float(coef[1]),
                                                "intercept": float(coef[2]), "moved_thresh": a.moved_thresh}
out["test_meta"] = {"box": ys["box"][a.n_train:], "total": ys["total"][a.n_train:], "mode": modes[a.n_train:],
                    "moved": [int(m) for m in moved[a.n_train:]]}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
