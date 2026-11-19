"""Render sample possible/impossible pairs to GIFs for visual review."""

from __future__ import annotations

import os

import imageio
import numpy as np

from .limiter import RenderLimiter
from .run import run_episode
from .sampler import Manifest, Template, make_twin, sample_episode
from .validate import validate_clip, validate_twins


def _label(frames: np.ndarray, color) -> np.ndarray:
    f = frames.copy()
    f[:, :3, :, :] = color
    return f


def preview(template: Template, manifest: Manifest, split: str, seeds, out_dir: str, backend: str = "mujoco",
            max_render_fps: float = 30.0, scale: int = 3) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    lim = RenderLimiter(os.path.join(out_dir, ".render_limiter"), max_render_fps)
    lines = []
    for seed in seeds:
        ep = sample_episode(template, manifest, split, seed)
        clip, diag = run_episode(ep, backend, lim)
        v = validate_clip(clip, diag)
        panels = [_label(clip.frames, (40, 200, 40))]
        status = f"possible: {'ok' if v.ok else '; '.join(v.reasons)}"
        if template.use_screen:
            tw = make_twin(ep, seed)
            tclip, tdiag = run_episode(tw, backend, lim)
            tv, pv = validate_clip(tclip, tdiag), validate_twins(clip, tclip)
            panels.append(_label(tclip.frames, (220, 40, 40)))
            status += f" | impossible ({tw.violation.kind}): {'ok' if tv.ok and pv.ok else '; '.join(tv.reasons + pv.reasons)}"
        frames = np.concatenate(panels, axis=2)
        frames = np.repeat(np.repeat(frames, scale, 1), scale, 2)
        m = ep.meta
        name = f"{ep.episode_id}_{m['n0']}{'+' if m['op'] == 'add' else '-'}{m['k']}={m['final']}.gif"
        imageio.mimsave(os.path.join(out_dir, name), list(frames), duration=1000 / ep.clip.fps, loop=0)
        lines.append(f"{name}: {status}")
    with open(os.path.join(out_dir, "preview.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return lines
