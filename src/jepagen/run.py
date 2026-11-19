"""Run an episode through a backend: step physics, apply the violation, render each frame."""

from __future__ import annotations

import numpy as np

from .backends.base import ClipResult
from .spec import Episode


def get_backend(name: str):
    if name == "mujoco":
        from .backends.mujoco_backend import MujocoBackend
        return MujocoBackend
    if name == "rust":
        from .backends.rust_backend import RustBackend
        return RustBackend
    raise ValueError(f"unknown backend {name!r}")


def run_episode(ep: Episode, backend: str, limiter) -> tuple[ClipResult, dict]:
    """Returns the clip and per-run physics diagnostics used by the validator."""
    b = get_backend(backend)(ep, limiter)
    try:
        frames, segs, states = [], [], []
        max_pen = 0.0
        v_t = ep.violation.t if ep.violation is not None else None
        applied = False
        for t in ep.clip.frame_times():
            if v_t is not None and not applied and v_t <= t:
                b.step_to(v_t)
                b.apply_violation()
                applied = True
            b.step_to(t)
            max_pen = max(max_pen, b.penetration())
            states.append(b.frame_state())
            rgb, seg = b.render()
            frames.append(rgb)
            segs.append(seg)
        if v_t is not None and not applied:
            raise RuntimeError("violation time past the last frame")
        clip = ClipResult(episode=ep, frames=np.stack(frames), seg=np.stack(segs), states=states, events=list(b.events))
        return clip, {"max_penetration": max_pen, "backend": backend}
    finally:
        b.close()
