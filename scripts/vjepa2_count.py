"""Stage 0 for a large pretrained flat encoder: does frozen V-JEPA 2 represent the objects in our scenes as
countable units?

Each fresh v4b clip's final frame (all objects at rest, box visible) becomes a static 16-frame clip at 256 px.
Token sets for the last temporal slot (16x16 = 256 tokens, the same grid as our oracle tokens):
  enc{L}  : V-JEPA 2 ViT-L context encoder, hidden state after layer L (6, 12, 18, 24)
  tgt     : the official V-JEPA 2 target encoder (the features its surprise is measured in)
  oracle  : positive control (count_readout.oracle_tokens)
The same two readouts as layer_probe.py (slot attention and slot-free transformer, both ~100% on oracle tokens)
are trained on frozen tokens to predict objects in the box and all objects in view.
GPU for the encoder only (ComputeGuard); readouts on CPU.

    python scripts/vjepa2_count.py --data DIR --out OUT
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F
import webdataset as wds

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from count_readout import SlotReadout, oracle_tokens  # noqa: E402
from jepaeval.guard import ComputeGuard  # noqa: E402
from jepaeval.vjepa import VJepaSurprise  # noqa: E402
from jepagen.shards import load_sample  # noqa: E402
from layer_probe import XfmrReadout, fit  # noqa: E402

LAYERS = (6, 12, 18, 24)
T_STATIC = 16


@torch.no_grad()
def tokens(sj: VJepaSurprise, frame) -> dict[str, torch.Tensor]:
    clip = torch.as_tensor(frame)[None].expand(T_STATIC, -1, -1, -1).contiguous()    # [T, H, W, 3] uint8
    x = sj.preprocess(clip.numpy()).unsqueeze(0).permute(0, 2, 1, 3, 4)               # HF layout [1, T, 3, H, W]
    G = sj.grid * sj.grid
    with sj.guard, torch.autocast("cuda", dtype=torch.float16):
        o = sj.model.encoder(x, output_hidden_states=True)
        t = sj.target_encoder(x).last_hidden_state
    hs = o.hidden_states
    if len(hs) != 25:
        raise RuntimeError(f"expected 25 hidden states (embedding + 24 layers), got {len(hs)}")
    out = {f"enc{L}": hs[L][0, -G:].float().cpu().half() for L in LAYERS}
    out["tgt"] = F.layer_norm(t.float(), (t.shape[-1],))[0, -G:].cpu().half()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--threads", type=int, default=40)
    ap.add_argument("--features", required=True, help="token cache (.pt): written if absent, read if present")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if os.path.exists(a.features):
        c = torch.load(a.features)
        if c["data"] != a.data or len(c["ys"]["box"]) != a.n:
            raise ValueError(f"{a.features} was built from {c['data']} with {len(c['ys']['box'])} clips")
        sets, ys = c["sets"], c["ys"]
    else:
        sj = VJepaSurprise(guard=ComputeGuard(duty=0.5))
        if sj.grid != 16:
            raise RuntimeError(f"V-JEPA 2 grid {sj.grid}; the oracle tokens assume 16")
        sets, ys = {}, {"box": [], "total": []}
        for s in wds.WebDataset(sorted(glob.glob(os.path.join(a.data, "train", "possible", "*.tar"))),
                                shardshuffle=False, empty_check=False):
            ep, frames, seg, state = load_sample(s)
            last = state["frames"][-1]
            in_box = {o["id"] for o in last["objects"] if o["region"] == "box"}
            for k, v in tokens(sj, frames[-1]).items():
                sets.setdefault(k, []).append(v)
            sets.setdefault("oracle", []).append(oracle_tokens(seg[-1], 16, in_box))
            ys["box"].append(int(ep["meta"]["final"]))
            ys["total"].append(int(last["count_per_region"]["box"] + last["count_per_region"]["hopper"]))
            if len(ys["box"]) % 250 == 0:
                print(f"  {len(ys['box'])} clips, guard {sj.guard.summary()}", flush=True)
            if len(ys["box"]) == a.n:
                break
        sets = {k: torch.stack(v) for k, v in sets.items()}
        ys = {k: torch.tensor(v) for k, v in ys.items()}
        torch.save({"data": a.data, "sets": sets, "ys": ys}, a.features)
        del sj
        torch.cuda.empty_cache()
    tr, te = slice(0, a.n_train), slice(a.n_train, len(ys["box"]))
    out = {"data": a.data, "n_train": a.n_train, "count": {}}
    for tname, yy in ys.items():
        chance = float((yy[te] == torch.bincount(yy[tr]).argmax()).float().mean() * 100)
        out[f"chance_{tname}"] = chance
        print(f"target {tname}: chance {chance:.1f}%", flush=True)
        for name in ["oracle", "tgt"] + [f"enc{L}" for L in LAYERS]:
            X = sets[name]
            d, n = X.shape[-1], X.shape[1]
            r = {"slots": fit(lambda: SlotReadout(d, n), X[tr], yy[tr], X[te], yy[te], a.epochs, 4e-4),
                 "xfmr": fit(lambda: XfmrReadout(d, n), X[tr], yy[tr], X[te], yy[te], a.epochs, 3e-4)}
            out["count"][f"{tname}/{name}"] = r
            print(f"  {name:<7} slots {r['slots']:5.1f}%  xfmr {r['xfmr']:5.1f}%", flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "vjepa2_count.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
