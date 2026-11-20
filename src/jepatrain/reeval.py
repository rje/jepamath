"""Re-run the checkpoint evaluation (reveal-locked twins incl. pooled score, linear probes) on saved weights."""

from __future__ import annotations

import argparse
import glob
import json
import os

import torch

from jepaeval.guard import ComputeGuard

from .evaluate import TrainedSurprise, evaluate_all
from .model import Encoder, Predictor


def load_run(run: str, weights: str, device: str = "cuda"):
    args = json.load(open(os.path.join(run, "args.json")))
    mk = lambda: Encoder(dim=args["dim"], depth=args["depth"], heads=args["dim"] // 64).to(device)
    enc, tgt = mk(), mk()
    pred = Predictor(enc, dim=args["pred_dim"], depth=args["pred_depth"], heads=args["pred_dim"] // 32).to(device)
    w = torch.load(os.path.join(run, weights), map_location=device)
    enc.load_state_dict(w["enc"]); pred.load_state_dict(w["pred"]); tgt.load_state_dict(w["tgt"])
    return enc, tgt, pred, w["step"]


def main():
    ap = argparse.ArgumentParser(prog="jepatrain.reeval")
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", nargs="*", help="default: every weights_*.pt not yet in reeval.jsonl")
    ap.add_argument("--heldout", default="/mnt/data/jepamath/heldout/v2_triples2000_128.pt")
    ap.add_argument("--duty", type=float, default=0.5)
    a = ap.parse_args()
    out = os.path.join(a.run, "reeval.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["weights"] for l in open(out)}
    ws = a.weights or sorted(os.path.basename(p) for p in glob.glob(os.path.join(a.run, "weights_*.pt")))
    ws = [w for w in ws if w not in done]
    if not ws:
        raise SystemExit("nothing to evaluate")
    triples = torch.load(a.heldout, weights_only=False)
    guard = ComputeGuard(duty=a.duty)
    for w in ws:
        enc, tgt, pred, step = load_run(a.run, w)
        r = evaluate_all(TrainedSurprise(enc, tgt, pred, guard), triples)
        rec = {"weights": w, "eval_step": step, "reveal": r["reveal"], "probes": r["probes"]}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        rp, pl = r["reveal"]["post_reveal"], r["reveal"]["post_reveal_pooled"]
        print(f"[reeval {step}] token add/del {rp['by']['imp=add']['imp>ctl']['acc']:.0f}/"
              f"{rp['by']['imp=delete']['imp>ctl']['acc']:.0f} | pooled imp>ctl {pl['imp>ctl']['acc']:.0f} "
              f"add/del {pl['by']['imp=add']['imp>ctl']['acc']:.0f}/{pl['by']['imp=delete']['imp>ctl']['acc']:.0f}",
              flush=True)


if __name__ == "__main__":
    main()
