"""Frozen corresponding-patch endpoint for same-start additions, removals, and common pickups."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from jepaeval.vjepa import IMAGENET_MEAN, IMAGENET_STD


def intervals(values):
    values = np.asarray(values)
    rng = np.random.default_rng(111)
    sampled = np.concatenate(
        [values[rng.integers(len(values), size=(200, len(values)))].mean(1) for _ in range(50)]
    )
    return np.stack([values.mean(0), *np.percentile(sampled, (2.5, 97.5), axis=0)], axis=-1).tolist()


def readouts(scores):
    a = np.asarray(scores)
    signs = np.tile([1, -1], a.shape[0] // 2)
    m = (a[:, 1] - a[:, 0]) * signs
    return (m > 1e-12) + 0.5 * (np.abs(m) <= 1e-12)


@torch.inference_mode()
def scene(z, prefix, enc, tgt, pred, controls, collect_features=None):
    q = z[prefix + "queries"]
    alts = z[prefix + "alternatives"]
    tok = np.flatnonzero(z[prefix + "box"])
    hopper = z[prefix + "hopper"]
    ring = z[prefix + "ring"]
    device = next(enc.parameters()).device

    def tensor(x):
        v = torch.as_tensor(x, device=device).permute(0, 4, 1, 2, 3).float() / 255
        mean = v.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1)
        std = v.new_tensor(IMAGENET_STD).view(1, 3, 1, 1, 1)
        return (v - mean) / std

    images = np.concatenate([q[:, -4:], alts])
    padded = np.zeros((len(images), 32, *images.shape[2:]), dtype=np.uint8)
    padded[:, -4:] = images
    g = enc.grid**2
    ix = torch.as_tensor(7 * g + tok, device=device)
    rep = tgt(tensor(padded), ix.expand(len(images), -1)).float()
    rep = F.layer_norm(rep, (rep.shape[-1],))
    bank = rep[len(q) :]

    def compare(x):
        ds = (x[:, None] - bank[None]).abs().mean((-1, -2))
        return torch.stack([ds[:, :4].mean(1), ds[:, 4:].mean(1)], 1).cpu().tolist()

    masked = q.copy()
    masked[:, :, hopper] = np.median(q[0, 0][ring], axis=0).astype(q.dtype)
    out = {"visible": compare(rep[: len(q)])}
    timings = [("prefix", 4), ("carry", 16), ("landed", 24)]
    if controls:
        timings.extend([("still", 16), ("delayed", 24)])
    for name, ctx in timings:
        fr = masked[:, :ctx]
        if name == "still":
            fr = np.repeat(fr[:, -1:], ctx, axis=1)
        if name == "delayed":
            fr = np.concatenate([masked[:, np.arange(7) % 4], masked[:, : ctx - 7]], axis=1)
        n = ctx // 4 * g
        encoded = enc(tensor(fr))
        if collect_features is not None:
            collect_features(name, encoded)
        predout = pred(
            encoded,
            torch.arange(n, device=device).expand(len(q), -1),
            torch.arange(n, 8 * g, device=device).expand(len(q), -1),
        )
        out[name] = compare(predout[:, ix - n].float())
    return out


def score(root, split, out, step, enc, tgt, pred):
    folder = Path(root) / split
    mp = folder / "manifest.json"
    manifest = json.loads(mp.read_text())
    rows = []
    enc.eval()
    tgt.eval()
    pred.eval()
    try:
        for seed in manifest["seeds"]:
            path = folder / f"{seed}.npz"
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][path.name]:
                raise RuntimeError("evaluation input changed")
            with np.load(path) as z:
                rows.append(
                    {
                        "seed": seed,
                        "main": scene(z, "", enc, tgt, pred, True),
                        "fork": scene(z, "fork_", enc, tgt, pred, True),
                    }
                )
    finally:
        enc.train()
        pred.train()
    summary = {}
    for experiment in ("main", "fork"):
        summary[experiment] = {}
        for stage in rows[0][experiment]:
            hits = np.array([readouts(r[experiment][stage]) for r in rows])
            groups = hits.reshape(len(hits), -1, 2)
            both = (groups == 1).all(-1)
            summary[experiment][stage] = {
                "cells": intervals(hits),
                "pairs": intervals(groups.mean(-1)),
                "both": intervals(both),
            }
    result = dict(
        step=step,
        split=split,
        manifest=hashlib.sha256(mp.read_bytes()).hexdigest(),
        summary=summary,
        rows=rows,
    )
    p = Path(out) / f"placement-{split}-{step:06}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, allow_nan=False))
    tmp.replace(p)
    print(f"[placement {split} {step}] {summary['main']['carry']['pairs']}", flush=True)
    if split == "dev" and step == 0 and min(x[1] for x in summary["main"]["visible"]["cells"]) <= 0.75:
        raise RuntimeError("frozen parent failed development visible control")


def compare_runs(control, treatment):
    c, t = [json.loads(Path(p).read_text()) for p in (control, treatment)]
    if (c["manifest"], c["step"], c["split"]) != (t["manifest"], t["step"], t["split"]):
        raise ValueError("unpaired results")
    out = {}
    for exp in ("main", "fork"):
        for stage in t["summary"][exp]:
            diffs = []
            for a, b in zip(c["rows"], t["rows"], strict=True):
                if a["seed"] != b["seed"]:
                    raise ValueError("unpaired seeds")
                delta = (readouts(b[exp][stage]) - readouts(a[exp][stage])).reshape(-1, 2)
                diffs.append(delta.mean(1))
            out[f"{exp}_{stage}"] = intervals(diffs)
    return out
