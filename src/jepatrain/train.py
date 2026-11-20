"""Track B: train a small video JEPA from scratch on possible v2 clips, under the GPU compute guard.

Masks (shared across a batch, resampled every step):
  prefix  (p_prefix)  : context = the first c time slots, target = the rest (the reveal-locked test's shape)
  short tube          : 8 spatial blocks of ~15% area, masked over all slots (V-JEPA short-range)
  long tube           : 2 spatial blocks of ~70% area, masked over all slots (V-JEPA long-range)
Loss: L1 between predictor output and the layer-normed EMA target encoder, on target tokens.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from jepaeval.guard import ComputeGuard
from jepaeval.vjepa import IMAGENET_MEAN, IMAGENET_STD

from .data import PossibleClips
from .evaluate import TrainedSurprise, evaluate_all
from .model import Encoder, Predictor


def tube_mask(rng: random.Random, grid: int, n_blocks: int, scale: float) -> torch.Tensor:
    m = torch.zeros(grid, grid, dtype=torch.bool)
    for _ in range(n_blocks):
        area = scale * grid * grid
        ar = math.exp(rng.uniform(math.log(0.75), math.log(1.5)))
        h = max(1, min(grid, round(math.sqrt(area * ar))))
        w = max(1, min(grid, round(math.sqrt(area / ar))))
        r, c = rng.randrange(grid - h + 1), rng.randrange(grid - w + 1)
        m[r:r + h, c:c + w] = True
    return m.flatten()


def sample_masks(rng: random.Random, slots: int, grid: int, p_prefix: float):
    G, N = grid * grid, slots * grid * grid
    u = rng.random()
    if u < p_prefix:
        c = rng.randint(1, slots - 1)
        return "prefix", torch.arange(c * G), torch.arange(c * G, N)
    kind, nb, sc = ("short", 8, 0.15) if u < p_prefix + (1 - p_prefix) / 2 else ("long", 2, 0.7)
    while True:
        m = tube_mask(rng, grid, nb, sc)
        if 0.1 * G <= (~m).sum() and m.sum() > 0:
            break
    tgt = m.repeat(slots)
    idx = torch.arange(N)
    return kind, idx[~tgt], idx[tgt]


def main():
    ap = argparse.ArgumentParser(prog="jepatrain")
    ap.add_argument("--data", default="/mnt/data/jepamath/v2")
    ap.add_argument("--heldout", default="/mnt/data/jepamath/heldout/v2_triples2000_128.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--ema", type=float, nargs=2, default=(0.996, 1.0))
    ap.add_argument("--p-prefix", type=float, default=0.5)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--pred-dim", type=int, default=256)
    ap.add_argument("--pred-depth", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--duty", type=float, default=0.5)
    ap.add_argument("--power-ceiling", type=float, default=200.0)
    ap.add_argument("--temp-ceiling", type=float, default=83.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = "cuda"
    guard = ComputeGuard(duty=a.duty, temp_ceiling_c=a.temp_ceiling, power_ceiling_w=a.power_ceiling)

    enc = Encoder(dim=a.dim, depth=a.depth, heads=a.dim // 64).to(dev)
    pred = Predictor(enc, dim=a.pred_dim, depth=a.pred_depth, heads=a.pred_dim // 32).to(dev)
    tgt = copy.deepcopy(enc).requires_grad_(False)
    params = [p for m in (enc, pred) for p in m.parameters()]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0}],
                            lr=a.lr, betas=(0.9, 0.95))
    step = 0
    ck_path = os.path.join(a.out, "last.pt")
    if a.resume:
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        enc.load_state_dict(ck["enc"]); pred.load_state_dict(ck["pred"]); tgt.load_state_dict(ck["tgt"])
        opt.load_state_dict(ck["opt"]); step = ck["step"]; rng.setstate(ck["rng"])
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    n_params = sum(p.numel() for p in params)
    print(f"[train] encoder+predictor {n_params / 1e6:.1f}M params, tokens/clip {enc.slots * enc.tokens_per_slot}", flush=True)

    # start the loader workers before the held-out set is in memory, so forked workers don't carry it
    with open(a.heldout + ".ids.json") as f:
        exclude = set(json.load(f))
    loader = torch.utils.data.DataLoader(PossibleClips(a.data, exclude, seed=a.seed + step), batch_size=a.batch,
                                         num_workers=a.workers, pin_memory=True, persistent_workers=True,
                                         prefetch_factor=4)
    it = iter(loader)
    triples = torch.load(a.heldout, weights_only=False)
    if {t["seed_id"] for t in triples} != exclude:
        raise RuntimeError("held-out ids file does not match the held-out triples")
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1, 1)
    sj = TrainedSurprise(enc, tgt, pred, guard)
    logf = open(os.path.join(a.out, "log.jsonl"), "a")
    acc = {}
    t_last = time.monotonic()

    def lr_at(s):
        if s < a.warmup:
            return a.lr * s / a.warmup
        q = (s - a.warmup) / max(1, a.steps - a.warmup)
        return 1e-6 + (a.lr - 1e-6) * 0.5 * (1 + math.cos(math.pi * q))

    def do_eval():
        t0 = time.monotonic()
        r = evaluate_all(sj, triples)
        torch.save(r["raw"], os.path.join(a.out, f"eval_raw_{step:06d}.pt"))
        rec = {"eval_step": step, "reveal": r["reveal"], "probes": r["probes"], "eval_s": round(time.monotonic() - t0, 1)}
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        rv = r["reveal"]["post_reveal"]
        print(f"[eval {step}] imp>ctl {rv['imp>ctl']['acc']:.1f} add {rv['by']['imp=add']['imp>ctl']['acc']:.1f} "
              f"delete {rv['by']['imp=delete']['imp>ctl']['acc']:.1f} | probes {r['probes']}", flush=True)
        torch.save({"enc": enc.state_dict(), "pred": pred.state_dict(), "tgt": tgt.state_dict(), "step": step},
                   os.path.join(a.out, f"weights_{step:06d}.pt"))

    if step == 0:
        do_eval()
    enc.train(); pred.train()
    while step < a.steps:
        clips = next(it).to(dev, non_blocking=True)
        kind, ctx, tg = sample_masks(rng, enc.slots, enc.grid, a.p_prefix)
        B = clips.shape[0]
        ctx, tg = ctx.to(dev).unsqueeze(0).expand(B, -1), tg.to(dev).unsqueeze(0).expand(B, -1)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with guard:
            x = (clips.permute(0, 4, 1, 2, 3).float() / 255 - mean) / std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    h = tgt(x).float()
                    h = F.layer_norm(h, (h.size(-1),))
                    ht = torch.gather(h, 1, tg[..., None].expand(-1, -1, h.shape[-1]))
                z = enc(x, keep=ctx)
                p = pred(z, ctx, tg).float()
                loss = (p - ht).abs().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            m = a.ema[0] + (a.ema[1] - a.ema[0]) * step / a.steps
            with torch.no_grad():
                torch._foreach_mul_(list(tgt.parameters()), m)
                torch._foreach_add_(list(tgt.parameters()), list(enc.parameters()), alpha=1 - m)
        step += 1
        for k, v in ((f"loss_{kind}", float(loss)), ("gnorm", float(gn)),
                     ("tgt_std", float(h.std(0).mean())), ("pred_std", float(p.std(0).mean()))):
            acc.setdefault(k, []).append(v)
        if step % a.log_every == 0:
            now = time.monotonic()
            rec = {"step": step, "lr": lr_at(step), "clips_per_s": round(a.log_every * B / (now - t_last), 1),
                   **{k: sum(v) / len(v) for k, v in acc.items()}, "guard": guard.summary()}
            t_last, acc = now, {}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            if step % (a.log_every * 10) == 0:
                print(f"[train] {json.dumps(rec)}", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            torch.save({"enc": enc.state_dict(), "pred": pred.state_dict(), "tgt": tgt.state_dict(),
                        "opt": opt.state_dict(), "step": step, "rng": rng.getstate()}, ck_path)
            do_eval()


if __name__ == "__main__":
    main()
