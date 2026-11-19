"""Run the same episodes on two backends and compare semantics (not pixels or trajectories)."""

from __future__ import annotations

import os

import numpy as np

from .limiter import RenderLimiter
from .run import run_episode
from .sampler import Manifest, Template, make_twin, sample_episode
from .validate import validate_clip


def _semantics(clip):
    counts = [tuple(sorted(s.count_per_region.items())) for s in clip.states]
    events = [(e["event"], e["object_id"]) for e in clip.events]
    return counts, events


def crosscheck(template: Template, manifest: Manifest, split: str, seeds, out_dir: str,
               backends=("mujoco", "rust"), max_render_fps: float = 30.0) -> list[dict]:
    lim = RenderLimiter(os.path.join(out_dir, ".render_limiter"), max_render_fps)
    rows = []
    for seed in seeds:
        ep = sample_episode(template, manifest, split, seed)
        eps = [ep] + ([make_twin(ep, seed)] if template.use_screen else [])
        for e in eps:
            res = {}
            for b in backends:
                clip, diag = run_episode(e, b, lim)
                res[b] = (clip, validate_clip(clip, diag))
            (ca, va), (cb, vb) = res[backends[0]], res[backends[1]]
            sa, sb = _semantics(ca), _semantics(cb)
            frame_mismatch = [i for i, (x, y) in enumerate(zip(sa[0], sb[0])) if x != y]
            rows.append({
                "episode_id": e.episode_id, "valid": {backends[0]: va.ok, backends[1]: vb.ok},
                "reasons": {backends[0]: va.reasons, backends[1]: vb.reasons},
                "count_frames_mismatched": len(frame_mismatch), "first_mismatch_frame": frame_mismatch[0] if frame_mismatch else None,
                "events_match": sa[1] == sb[1],
                "final_box": {backends[0]: ca.states[-1].count_per_region["box"], backends[1]: cb.states[-1].count_per_region["box"]},
                "pixel_mae": float(np.abs(ca.frames.astype(np.int16) - cb.frames).mean()),
            })
    return rows
