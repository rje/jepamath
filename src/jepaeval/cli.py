"""jepaeval: frozen V-JEPA 2 surprise evaluations (Track A smoke test, Garrido IntPhys reproduction)."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .guard import ComputeGuard
from .vjepa import MODEL_ID, VJepaSurprise


def main():
    ap = argparse.ArgumentParser(prog="jepaeval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("intphys", "twins"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="directory for metrics.json and raw.pt")
        p.add_argument("--model", default=MODEL_ID)
        p.add_argument("--duty", type=float, default=0.5, help="max fraction of wall time the GPU is busy")
        p.add_argument("--power-ceiling", type=float, default=200.0)
        p.add_argument("--temp-ceiling", type=float, default=83.0)
        p.add_argument("--window-batch", type=int, default=16)
    sub.choices["intphys"].add_argument("--root", required=True, help="IntPhys dev dir holding O1 O2 O3")
    sub.choices["intphys"].add_argument("--max-scenes", type=int, default=None)
    sub.choices["twins"].add_argument("--data", required=True)
    sub.choices["twins"].add_argument("--split", default="train")
    sub.choices["twins"].add_argument("--n", type=int, default=500)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    guard = ComputeGuard(duty=a.duty, temp_ceiling_c=a.temp_ceiling, power_ceiling_w=a.power_ceiling)
    sj = VJepaSurprise(a.model, guard=guard, window_batch=a.window_batch)
    t0 = time.monotonic()
    if a.cmd == "intphys":
        from .intphys import evaluate
        r = evaluate(a.root, sj, max_scenes=a.max_scenes)
    else:
        from .twins import evaluate
        r = evaluate(a.data, a.split, sj, a.n)
    r["metrics"]["guard"] = guard.summary()
    r["metrics"]["seconds"] = round(time.monotonic() - t0, 1)
    r["metrics"]["model"] = a.model
    torch.save(r["raw"], os.path.join(a.out, "raw.pt"))
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(r["metrics"], f, indent=1)
    print(json.dumps(r["metrics"], indent=1))


if __name__ == "__main__":
    main()
