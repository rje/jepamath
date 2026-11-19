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


def run_episodes_batched(eps: list[Episode], limiter) -> list[tuple[ClipResult, dict]]:
    """Rust engine only: runs N episodes in lockstep and renders each frame of all N in ONE GPU
    submission (one limiter slot). Frame i of episode j is identical to `run_episode`'s output."""
    import jepamath_engine

    from .backends.rust_backend import RustBackend

    if not eps:
        raise ValueError("no episodes")
    if len(eps) > jepamath_engine.max_batch():
        raise ValueError(f"batch of {len(eps)} exceeds engine max {jepamath_engine.max_batch()}")
    clip0 = eps[0].clip
    if any(e.clip != clip0 for e in eps):
        raise ValueError("batched episodes must share the clip spec (frame times and size)")
    bs = [RustBackend(e, limiter) for e in eps]
    n = len(eps)
    frames = [[] for _ in range(n)]
    segs = [[] for _ in range(n)]
    states = [[] for _ in range(n)]
    max_pen = [0.0] * n
    applied = [e.violation is None for e in eps]
    for t in clip0.frame_times():
        for j, (e, b) in enumerate(zip(eps, bs)):
            if not applied[j] and e.violation.t <= t:
                b.step_to(e.violation.t)
                b.apply_violation()
                applied[j] = True
            b.step_to(t)
            max_pen[j] = max(max_pen[j], b.penetration())
            states[j].append(b.frame_state())
        limiter.acquire()
        rgb, seg = jepamath_engine.render_batch([b.eng for b in bs])
        for j in range(n):
            frames[j].append(rgb[j])
            segs[j].append(seg[j])
    if not all(applied):
        raise RuntimeError("violation time past the last frame")
    return [(ClipResult(episode=e, frames=np.stack(frames[j]), seg=np.stack(segs[j]), states=states[j],
                        events=list(bs[j].events)),
             {"max_penetration": max_pen[j], "backend": "rust"}) for j, e in enumerate(eps)]
