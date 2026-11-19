"""Backend protocol and the shared mechanics rules every backend must implement identically.

The rules here (object resolution, claw waypoint schedule, screen motion, region
classification) are the semantic contract between the MuJoCo and Rust backends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..spec import Episode, ScreenMove, Transfer, Tray

# Segmentation label conventions (uint8 per pixel)
SEG_BG = 0
SEG_CLAW = 254
SEG_SCREEN = 253
MAX_OBJECTS = 250  # object id i -> label i + 1

CLAW_PAD_HALF = (0.022, 0.022, 0.006)
CLAW_ROD_RADIUS = 0.006
CLAW_ROD_LEN = 0.6
RESERVE_Z = -3.0
CLAW_RETREAT_S = 0.6
RELEASE_GAP = 0.008
DROP_GRID = 9
SCREEN_DOWN_TOP_Z = -0.06  # fully under the 5 cm table top

# Transfer segments as fractions of the transfer's duration
SEGMENTS = (
    ("approach", 0.20),   # current claw pos -> above the object, at travel height
    ("descend", 0.17),    # -> touching the object's top
    ("grasp", 0.05),      # weld on at segment start, hold
    ("lift", 0.14),       # -> travel height
    ("carry", 0.20),      # -> above drop point
    ("lower", 0.16),      # -> release height
    ("settle", 0.04),     # hold still so the carried object stops swinging
    ("release", 0.04),    # weld off at segment start, hold
)


@dataclass
class FrameState:
    t: float
    count_per_region: dict[str, int]
    objects: list[dict]
    claw_pos: tuple[float, float, float]
    screen_top_z: float
    active_action: str | None


@dataclass
class ClipResult:
    episode: Episode
    frames: np.ndarray        # uint8 [T, H, W, 3]
    seg: np.ndarray           # uint8 [T, H, W]
    states: list[FrameState]
    events: list[dict] = field(default_factory=list)  # grasp/release/violation log


class Backend(Protocol):
    name: str

    def __init__(self, episode: Episode, limiter) -> None: ...
    def step_to(self, t: float) -> None: ...
    def apply_violation(self) -> None: ...
    def frame_state(self) -> FrameState: ...
    def render(self) -> tuple[np.ndarray, np.ndarray]: ...
    def close(self) -> None: ...


def smoothstep(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def lerp3(a, b, s):
    return tuple(float(a[i] + (b[i] - a[i]) * s) for i in range(3))


def screen_top_z(ep: Episode, t: float) -> float:
    """Height of the screen's top edge. Down: just below the table surface. Up: `height`."""
    down, up = SCREEN_DOWN_TOP_Z, ep.scene.screen.height
    z = down
    for a in sorted((a for a in ep.actions if isinstance(a, ScreenMove)), key=lambda a: a.t_start):
        if t < a.t_start:
            break
        s = smoothstep((t - a.t_start) / (a.t_end - a.t_start))
        z = down + (up - down) * s if a.kind == "screen_raise" else up + (down - up) * s
    return z


def in_tray(tray: Tray, p) -> bool:
    hx, hy = tray.inner_size[0] / 2, tray.inner_size[1] / 2
    return (abs(p[0] - tray.center[0]) <= hx and abs(p[1] - tray.center[1]) <= hy
            and -0.005 <= p[2] <= tray.wall_height + 0.25)


def classify_region(ep: Episode, p, held: bool) -> str:
    if held:
        return "claw"
    if p[2] < RESERVE_Z + 1.0:
        return "reserve"
    if in_tray(ep.scene.box, p):
        return "box"
    if in_tray(ep.scene.hopper, p):
        return "hopper"
    return "table"


def z_extent(shape: str, size, quat) -> float:
    """Half-height of the object's world-axis-aligned extent along z, given orientation (w,x,y,z)."""
    w, x, y, z = quat
    # third row of the rotation matrix
    r20, r21, r22 = 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
    if shape == "sphere":
        return float(size[0])
    if shape == "box":
        return float(abs(r20) * size[0] + abs(r21) * size[1] + abs(r22) * size[2])
    r, h = size
    return float(abs(r22) * h + r * math.sqrt(max(0.0, 1 - r22 * r22)))


def horiz_radius(shape: str, size) -> float:
    if shape == "box":
        return float(math.hypot(size[0], size[1]))
    return float(size[0])


def pick_topmost(ep: Episode, region: str, poses: dict[int, tuple], held: set[int]) -> int:
    """Shared rule: the object in `region` whose top is highest; ties -> lowest id."""
    shapes = {o.id: o for o in ep.objects}
    best = None
    for oid in sorted(poses):
        if oid in held:
            continue
        pos, quat = poses[oid]
        if classify_region(ep, pos, False) != region:
            continue
        top = pos[2] + z_extent(shapes[oid].shape, shapes[oid].size, quat)
        if best is None or top > best[0] + 1e-9:
            best = (top, oid)
    if best is None:
        raise RuntimeError(f"no object in region {region!r} to pick")
    return best[1]


def choose_drop(ep: Episode, dst: Tray, dst_region: str, oid: int, drop_xy, poses) -> tuple[float, float, float]:
    """Shared drop rule (claw releases and hidden 'add' violations). Candidates: a DROP_GRID x
    DROP_GRID grid over the tray interior, inset by the object's horizontal radius + 5 mm. Surface
    under a candidate = tray floor top, raised to the top of any other object in the tray whose
    centre is within the sum of horizontal radii. Pick the lowest surface (1 mm buckets), then
    nearest to drop_xy, then lowest grid index. Returns (x, y, surface_z)."""
    o = next(x for x in ep.objects if x.id == oid)
    shapes = {x.id: x for x in ep.objects}
    rh = horiz_radius(o.shape, o.size)
    others = []
    for other_id, (p, qq) in sorted(poses.items()):
        if other_id == oid or classify_region(ep, p, False) != dst_region:
            continue
        ot = shapes[other_id]
        others.append((p, horiz_radius(ot.shape, ot.size), p[2] + z_extent(ot.shape, ot.size, qq)))
    hx = dst.inner_size[0] / 2 - rh - 0.005
    hy = dst.inner_size[1] / 2 - rh - 0.005
    if hx <= 0 or hy <= 0:
        raise RuntimeError("object too large for destination tray")
    best = None
    for i in range(DROP_GRID):
        for j in range(DROP_GRID):
            x = dst.center[0] - hx + 2 * hx * i / (DROP_GRID - 1)
            y = dst.center[1] - hy + 2 * hy * j / (DROP_GRID - 1)
            surf = dst.wall_thickness
            for p, r, top in others:
                if math.hypot(p[0] - x, p[1] - y) < rh + r:
                    surf = max(surf, top)
            key = (round(surf / 0.001), math.hypot(x - drop_xy[0], y - drop_xy[1]), i * DROP_GRID + j)
            if best is None or key < best[0]:
                best = (key, (float(x), float(y), float(surf)))
    return best[1]


VIOLATION_ADD_GAP = 0.002


def violation_add_pos(ep: Episode, oid: int, pos_hint, poses) -> tuple[float, float, float]:
    """Where a hidden 'add' places its object: the drop rule over the box, resting just above the surface."""
    o = next(x for x in ep.objects if x.id == oid)
    x, y, surf = choose_drop(ep, ep.scene.box, "box", oid, (pos_hint[0], pos_hint[1]), poses)
    return (x, y, surf + z_extent(o.shape, o.size, (1.0, 0.0, 0.0, 0.0)) + VIOLATION_ADD_GAP)


class ClawController:
    """Claw target position over time. Waypoints are resolved lazily from live object poses
    at segment boundaries, so behaviour is defined by the rules above, not by precomputed paths.

    Calls back `grasp(oid)` / `release(oid)` at segment starts.
    """

    def __init__(self, ep: Episode, get_poses, grasp, release):
        self.ep = ep
        self.home = ep.physics.claw_home
        self.travel_z = ep.physics.claw_travel_z
        self.transfers = sorted((a for a in ep.actions if isinstance(a, Transfer)), key=lambda a: a.t_start)
        self.get_poses = get_poses
        self.grasp_cb, self.release_cb = grasp, release
        self.pos = tuple(self.home)
        self._seg_key = None
        self._seg_from = self.pos
        self._seg_to = self.pos
        self.target_oid: int | None = None
        self.held: int | None = None
        self.events: list[dict] = []
        self.drop: tuple[float, float, float] | None = None

    def _shape(self, oid):
        return next(o for o in self.ep.objects if o.id == oid)

    def _resolve_target(self, i: int, name: str, tr: Transfer):
        poses = self.get_poses()
        if name == "approach":
            self.target_oid = pick_topmost(self.ep, tr.src, poses, set())
            p, _ = poses[self.target_oid]
            return (p[0], p[1], self.travel_z)
        if name == "descend":
            p, q = poses[self.target_oid]
            o = self._shape(self.target_oid)
            return (p[0], p[1], p[2] + z_extent(o.shape, o.size, q) + CLAW_PAD_HALF[2] + 0.001)
        if name in ("grasp", "settle", "release"):
            return self.pos
        if name == "lift":
            return (self.pos[0], self.pos[1], self.travel_z)
        if name == "carry":
            self.drop = self._choose_drop(tr, poses)
            return (self.drop[0], self.drop[1], self.travel_z)
        if name == "lower":
            # Nominal claw-above-centre offset, exactly as set up at grasp (see "descend"); not the
            # live offset, which depends on how much each engine's joint sags while moving.
            o = self._shape(self.held)
            _, q = poses[self.held]
            ext = z_extent(o.shape, o.size, q)
            claw_offset = ext + CLAW_PAD_HALF[2] + 0.001
            return (self.drop[0], self.drop[1], self.drop[2] + RELEASE_GAP + ext + claw_offset)
        raise AssertionError(name)

    def _choose_drop(self, tr: Transfer, poses) -> tuple[float, float, float]:
        dst = self.ep.scene.box if tr.dst == "box" else self.ep.scene.hopper
        return choose_drop(self.ep, dst, tr.dst, self.held, tr.drop_xy, poses)

    def update(self, t: float) -> tuple[float, float, float]:
        for i, tr in enumerate(self.transfers):
            if tr.t_start <= t < tr.t_end:
                u = (t - tr.t_start) / (tr.t_end - tr.t_start)
                acc = 0.0
                for j, (name, frac) in enumerate(SEGMENTS):
                    if u < acc + frac or j == len(SEGMENTS) - 1:
                        key = (i, j)
                        if key != self._seg_key:
                            self._seg_key = key
                            self._seg_from = self.pos
                            if name == "grasp":
                                self.held = self.target_oid
                                self.grasp_cb(self.held)
                                self.events.append({"t": t, "event": "grasp", "object_id": self.held})
                            if name == "release":
                                self.release_cb(self.held)
                                self.events.append({"t": t, "event": "release", "object_id": self.held})
                                self.held = None
                            self._seg_to = self._resolve_target(i, name, tr)
                        s = smoothstep((u - acc) / frac)
                        self.pos = lerp3(self._seg_from, self._seg_to, s)
                        return self.pos
                    acc += frac
        # idle: after a transfer ends, retreat home over CLAW_RETREAT_S
        done = [tr for tr in self.transfers if tr.t_end <= t]
        if self.held is not None and done:
            raise RuntimeError("transfer ended with an object still held")
        if done:
            last = done[-1]
            key = ("retreat", len(done))
            if key != self._seg_key:
                self._seg_key = key
                self._seg_from = self.pos
            s = smoothstep((t - last.t_end) / CLAW_RETREAT_S)
            self.pos = lerp3(self._seg_from, self.home, s)
        return self.pos
