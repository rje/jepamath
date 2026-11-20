"""IntPhys dev reproduction of Garrido et al. 2025 (validates the surprise pipeline).

Protocol copied from facebookresearch/jepa-intuitive-physics (evals/intuitive_physics):
frames 0..98 every 2nd (49 frames), 16-frame windows with stride 2, context lengths 2..10,
L1 on layer-normed targets. Each scene has 4 movies (2 possible, 2 impossible); movies are paired
by their first differing frame, as in their get_breaking_points/get_matches.
Relative accuracy = fraction of pairs where the impossible movie has the higher surprise.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from PIL import Image

from .vjepa import VJepaSurprise

FRAME_STEP = 2   # Garrido's default


def load_movie(path: str, frame_step: int = FRAME_STEP) -> np.ndarray:
    frames = sorted(os.listdir(os.path.join(path, "scene")))
    out = []
    n = 99 // frame_step
    for f in frames[: n * frame_step: frame_step]:
        im = Image.open(os.path.join(path, "scene", f))
        if im.mode not in ("RGB", "RGBA"):
            raise ValueError(f"unexpected image mode {im.mode} in {path}/{f}")
        a = np.asarray(im)
        if a.shape[-1] == 4:
            if not (a[..., 3] == 255).all():
                raise ValueError(f"non-opaque alpha in {path}/{f}")
            a = a[..., :3]
        out.append(a)
    return np.stack(out)


def is_possible(path: str) -> bool:
    with open(os.path.join(path, "status.json")) as f:
        return bool(json.load(f)["header"]["is_possible"])


def _matches(movies: list[np.ndarray]) -> list[tuple[int, int]]:
    """Pair the 4 movies of a scene: movie 0 pairs with the one it shares the longest prefix with."""
    bps = []
    for j in (1, 2, 3):
        diff = np.nonzero((movies[0] != movies[j]).reshape(movies[0].shape[0], -1).any(1))[0]
        bps.append(diff[0] if len(diff) else movies[0].shape[0])
    j = int(np.argmax(bps)) + 1
    rest = [k for k in (1, 2, 3) if k != j]
    return [(0, j), (rest[0], rest[1])]


def evaluate(root: str, sj: VJepaSurprise, blocks=("O1", "O2", "O3"), context_lengths=(2, 4, 6, 8, 10),
             max_scenes: int | None = None, frame_step: int = FRAME_STEP, log=print) -> dict:
    res = {}
    raw = {}
    for block in blocks:
        bdir = os.path.join(root, block)
        scenes = sorted(os.listdir(bdir))[:max_scenes]
        pairs = []   # (loss_possible [C, W], loss_impossible [C, W])
        for si, scene in enumerate(scenes):
            paths = [os.path.join(bdir, scene, str(k)) for k in (1, 2, 3, 4)]
            movies = [load_movie(p, frame_step) for p in paths]
            labels = [is_possible(p) for p in paths]
            losses = [sj.window_losses(sj.preprocess(m), context_lengths) for m in movies]
            for a, b in _matches(movies):
                if labels[a] == labels[b]:
                    raise RuntimeError(f"{block}/{scene}: matched movies {a + 1},{b + 1} have the same label")
                pos, imp = (a, b) if labels[a] else (b, a)
                pairs.append((losses[pos], losses[imp]))
            if (si + 1) % 10 == 0:
                log(f"[intphys] {block} {si + 1}/{len(scenes)} scenes, guard {sj.guard.summary()}")
        P = torch.stack([p for p, _ in pairs])   # [pairs, C, W]
        I = torch.stack([i for _, i in pairs])
        raw[block] = {"possible": P, "impossible": I}
        for ci, c in enumerate(context_lengths):
            res[f"{block}/ctx{c}"] = {
                "rel_acc_avg": float((I[:, ci].mean(-1) > P[:, ci].mean(-1)).float().mean() * 100),
                "rel_acc_max": float((I[:, ci].max(-1).values > P[:, ci].max(-1).values).float().mean() * 100),
                "n_pairs": len(pairs)}
    return {"metrics": res, "raw": raw}
