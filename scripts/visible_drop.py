"""Visible placement test: does the model predict where a claw-carried object lands when it watches the drop?

On unseen visible-only clips (no screen), take the last transfer, if it puts an object into the box and nothing
happens after it. A counterfactual clip is the real one with the dropped object painted out of every frame after
the release (its pixels replaced by the same pixels from the first frame, before the claw moved anything). The
two clips are identical up to the release, so with the context ending earlier the model makes one prediction for
both, and the question is which ending it is closer to inside the painted region (the tokens that differ):

  held   : context ends while the object is still in the claw above the box. Expecting the object means
           predicting the drop: the analogue of a hidden placement, without the occlusion.
  landed : context ends after the object has come to rest in view. Expecting it is just remembering a visible
           object (sanity check: should be near 100%).

The painted-out clip keeps the object's shadow, which may itself look odd; that affects both conditions alike,
and step-0 weights give the untrained baseline.

    CUDA_VISIBLE_DEVICES= python scripts/visible_drop.py RUN_DIR --weights W [W ...] --data DIR --out OUT
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
import torch.nn.functional as F
import webdataset as wds

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from jepagen.shards import load_sample  # noqa: E402
from jepatrain.data import downsample  # noqa: E402
from jepatrain.reeval import load_run  # noqa: E402
from locate_surprise import MEAN, STD, token_mask  # noqa: E402

FPS, SETTLE_S = 4.0, 0.5


def cases(data: str, n: int) -> list[dict]:
    """Clips whose final action is a transfer into the box; with release time, object id and both endings."""
    out = []
    for s in wds.WebDataset(sorted(glob.glob(os.path.join(data, "train", "possible", "*.tar"))), shardshuffle=False,
                            empty_check=False):
        ep, frames, seg, state = load_sample(s)
        if ep["meta"]["visibility"] != "visible":
            raise ValueError(f"{ep['episode_id']}: not a visible clip")
        acts = sorted(ep["actions"], key=lambda a: a["t_start"])
        last = acts[-1]
        if last["kind"] != "transfer" or last["dst"] != "box":
            continue
        rel = [e for e in state["events"] if e["event"] == "release" and last["t_start"] <= e["t"] <= last["t_end"]]
        grasp = [e for e in state["events"] if e["event"] == "grasp" and last["t_start"] <= e["t"] <= last["t_end"]]
        if len(rel) != 1 or len(grasp) != 1:
            raise ValueError(f"{ep['episode_id']}: expected one grasp/release in the last transfer, got {grasp} {rel}")
        oid, t_rel = rel[0]["object_id"], rel[0]["t"]
        f_rel = math.ceil(t_rel * FPS)                  # first frame after the release
        cf = frames.copy()
        for f in range(f_rel, len(frames)):
            m = seg[f] == oid + 1
            cf[f][m] = frames[0][m]
        pos = [np.array(fs["objects"][oid]["pos"]) for fs in state["frames"]]
        out.append({"id": ep["episode_id"], "op": ep["meta"]["op"], "src": last["src"], "oid": oid, "t_rel": t_rel,
                    "pos": pos,
                    "t_grasp": grasp[0]["t"], "frames": frames, "cf": cf})
        if len(out) == n:
            break
    return out


@torch.no_grad()
def score(enc, tgt, pred, c: dict, cond: str, res: int, device: str) -> dict | None:
    tb, G, T = enc.tubelet, enc.grid * enc.grid, c["frames"].shape[0]
    settle = math.ceil((c["t_rel"] + SETTLE_S) * FPS)          # first frame with the object at rest
    if cond == "held":
        ctx = int(c["t_rel"] * FPS) // tb * tb                 # every context frame is before the release
        if ctx < tb or (ctx - 1) / FPS <= c["t_grasp"]:
            return None                                        # the claw must already hold it in context
    else:
        ctx = math.ceil(settle / tb) * tb
    first = max(0, math.ceil((settle - ctx) / tb))             # first scored slot: object at rest
    if ctx >= T or ctx + first * tb >= T:
        return None
    fr, cf = downsample(c["frames"], res), downsample(c["cf"], res)
    # held: the two clips share their context, so the one prediction stands for both. landed: the context is the
    # real clip's (the object visibly at rest), which the counterfactual lacks by construction.
    if cond == "held" and not np.array_equal(fr[:ctx], cf[:ctx]):
        raise ValueError(f"{c['id']}: counterfactual differs inside the context")
    x = torch.stack([torch.from_numpy(fr), torch.from_numpy(cf)])
    x = ((x.permute(0, 4, 1, 2, 3).float() / 255 - MEAN) / STD).to(device)
    n_ctx, N = ctx // tb * G, T // tb * G
    ar = lambda a, b: torch.arange(a, b, device=device).unsqueeze(0)
    p = pred(enc(x[:1, :, :ctx]), ar(0, n_ctx), ar(n_ctx, N)).float()[0]
    h = tgt(x).float()
    h = F.layer_norm(h, (h.size(-1),))[:, n_ctx:]
    S = (T - ctx) // tb
    err = (p[None] - h).abs().mean(-1).view(2, S, G)[:, first:].mean(1).cpu().numpy()
    post = slice(ctx + first * tb, T)
    M = token_mask(fr[post], cf[post], enc.patch, 30, 0.15)
    if not M.any():
        return None
    # lambda: where the prediction sits on the line from the counterfactual target (0) to the real one (1),
    # over the masked tokens of the scored slots. The binary "closer to" test is lambda > 0.5, which an
    # L2 predictor near the mean feature fails for objects even when it knows (objects are far from the mean).
    Mt = torch.from_numpy(M).to(device)
    pv, rv, cv = (z.view(S, G, -1)[first:][:, Mt] for z in (p, h[0], h[1]))
    lam = float(((pv - cv) * (rv - cv)).sum() / ((rv - cv) ** 2).sum())
    # how far (horizontally) the object still was from where it ends up, in the last context frame
    gap = float(np.linalg.norm(c["pos"][ctx - 1][:2] - c["pos"][-1][:2]))
    return {"d_real": float(err[0][M].mean()), "d_cf": float(err[1][M].mean()), "lam": lam, "gap_m": gap,
            "mask_tokens": int(M.sum()), "ctx": ctx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--data", required=True, help="generated visible-only clips (with seg and state)")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.device == "cpu" and torch.cuda.is_available():
        raise SystemExit("--device cpu: run with CUDA_VISIBLE_DEVICES=")
    torch.set_num_threads(a.threads)
    cs = cases(a.data, a.n)
    print(f"{len(cs)} clips end with a transfer into the box", flush=True)
    os.makedirs(a.out, exist_ok=True)
    for w in a.weights:
        enc, tgt, pred, step = load_run(a.run, w, device=a.device)
        for m in (enc, tgt, pred):
            m.eval()
        res = json.load(open(os.path.join(a.run, "args.json")))["res"]
        rows = {cond: [] for cond in ("held", "landed")}
        for c in cs:
            for cond in rows:
                r = score(enc, tgt, pred, c, cond, res, a.device)
                if r is not None:
                    rows[cond].append({"id": c["id"], "src": c["src"], **r})
        name = f"{os.path.basename(a.run.rstrip('/'))}_{step:06d}"
        summ = {}
        for cond, rs in rows.items():
            for lab, f in (("all", lambda r: True), ("from hopper", lambda r: r["src"] == "hopper"),
                           ("within box", lambda r: r["src"] == "box"),
                           ("above spot (<2cm)", lambda r: r["gap_m"] < 0.02),
                           ("en route (>=2cm)", lambda r: r["gap_m"] >= 0.02)):
                d = np.array([r["d_cf"] - r["d_real"] for r in rs if f(r)])
                if len(d) == 0:
                    continue
                lam = np.array([r["lam"] for r in rs if f(r)])
                summ[f"{cond}/{lab}"] = {"n": len(d), "expects_object": float((d > 0).mean() * 100),
                                         "lam_mean": float(lam.mean()),
                                         "lam_sem": float(lam.std(ddof=1) / math.sqrt(len(lam))),
                                         "diff": float(d.mean()), "sem": float(d.std(ddof=1) / math.sqrt(len(d)))}
        with open(os.path.join(a.out, f"visdrop_{name}.json"), "w") as f:
            json.dump({"run": a.run, "step": step, "summary": summ, "rows": rows}, f)
        print(name, flush=True)
        for k, v in summ.items():
            print(f"  {k:<20} n={v['n']:4d}  lambda {v['lam_mean']:+.3f}±{v['lam_sem']:.3f}  "
                  f"closer to real {v['expects_object']:5.1f}%  "
                  f"diff {v['diff']:+.4f}±{v['sem']:.4f}", flush=True)


if __name__ == "__main__":
    main()
