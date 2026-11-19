"""Rust engine backend (rapier3d physics + wgpu/Vulkan rendering), via the jepamath_engine extension."""

from __future__ import annotations

import json

import numpy as np

import jepamath_engine

from ..spec import Episode
from .base import FrameState


class RustBackend:
    name = "rust"

    def __init__(self, episode: Episode, limiter):
        self.ep = episode
        self.limiter = limiter
        self.eng = jepamath_engine.Engine(episode.to_json())

    @property
    def events(self) -> list[dict]:
        return json.loads(self.eng.events())

    @property
    def t(self) -> float:
        return self.eng.time()

    def step_to(self, t: float) -> None:
        self.eng.step_to(t)

    def apply_violation(self) -> None:
        self.eng.apply_violation()

    def frame_state(self) -> FrameState:
        d = json.loads(self.eng.frame_state())
        for o in d["objects"]:
            for k in ("pos", "quat", "lin_vel", "ang_vel"):
                o[k] = tuple(o[k])
        d["claw_pos"] = tuple(d["claw_pos"])
        return FrameState(**d)

    def penetration(self) -> float:
        return self.eng.penetration()

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        # one GPU submission renders both colour and segmentation; it takes one limiter slot
        self.limiter.acquire()
        rgb, seg = self.eng.render()
        return rgb, seg

    def close(self) -> None:
        pass
