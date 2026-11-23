"""Concrete episode spec: every value fixed. Produced by the sampler, consumed by all backends.

Coordinates: metres, z up, table top at z = 0. Quaternions are (w, x, y, z).
Times are seconds from clip start (t = 0 is the first rendered frame; the
backend runs `physics.preroll` seconds of unrendered settling before it).
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SPEC_VERSION = "0.3"  # 0.3: glides

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
RGBA = tuple[float, float, float, float]
Shape = Literal["sphere", "box", "cylinder"]
# Regions an object can be in. "reserve" is an off-camera parking spot used only to
# hold the object an impossible twin adds, or to receive the object it deletes.
Region = Literal["box", "hopper", "reserve", "claw", "table"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Tray(_Strict):
    """Open-topped container: a floor plate plus four walls, sitting on the table."""

    name: Literal["box", "hopper"]
    center: Vec2
    inner_size: Vec2  # interior x, y extent
    wall_height: float
    wall_thickness: float
    color: RGBA


class Screen(_Strict):
    """Vertical occluder in front of the box. Rises out of a slot in the table.

    `raised` means it hides the box interior from the camera. The panel's plane is vertical;
    `yaw` rotates it about z (0 = panel in the x-z plane, facing -y).
    """

    center_x: float
    y: float
    yaw: float = 0.0
    width: float
    height: float
    thickness: float
    color: RGBA


class Camera(_Strict):
    pos: Vec3
    lookat: Vec3
    fovy: float  # degrees


class Light(_Strict):
    pos: Vec3
    dir: Vec3
    diffuse: Vec3
    castshadow: bool = True


class Scene(_Strict):
    table_size: Vec2
    table_color: RGBA
    background: Vec3
    ambient: Vec3
    box: Tray
    hopper: Tray
    screen: Screen
    camera: Camera
    lights: list[Light]


class ObjectSpec(_Strict):
    id: int
    shape: Shape
    # sphere: radius; box: half-extents; cylinder: (radius, half-height)
    size: tuple[float, ...]
    color: RGBA
    mass: float
    start_region: Literal["box", "hopper", "reserve"]
    start_pos: Vec3
    start_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @model_validator(mode="after")
    def _check_size(self) -> "ObjectSpec":
        n = {"sphere": 1, "box": 3, "cylinder": 2}[self.shape]
        if len(self.size) != n:
            raise ValueError(f"{self.shape} needs {n} size values, got {self.size}")
        return self


class ScreenMove(_Strict):
    kind: Literal["screen_raise", "screen_lower"]
    t_start: float
    t_end: float


class Transfer(_Strict):
    """The claw moves ONE object from `src` to `dst`.

    Which object is resolved at run time by a fixed rule shared by all backends:
    the object in `src` whose top is highest; ties broken by lowest id.
    """

    kind: Literal["transfer"] = "transfer"
    src: Literal["box", "hopper"]
    dst: Literal["box", "hopper"]
    t_start: float
    t_end: float
    drop_xy: Vec2  # release point inside dst (absolute coordinates)


Action = ScreenMove | Transfer


class Glide(_Strict):
    """Object `object_id` moves on its own across its tray floor to `to_xy` over [t_start, t_end]
    (smoothstep path from wherever it is at t_start). Spheres roll, other shapes slide upright.
    Not during a transfer or while the screen is up or moving (the twins must stay identical)."""

    object_id: int
    t_start: float
    t_end: float
    to_xy: Vec2


ViolationKind = Literal["add", "delete", "move", "swap"]
COUNT_CHANGING = ("add", "delete")   # impossible twins
COUNT_PRESERVING = ("move", "swap")  # control twins: an unexplained hidden change, same count


class Violation(_Strict):
    """Hidden change applied to a twin while the box is occluded.

    add:    reserve object `object_id` placed into the box; `pos` is a hint -- backends place it
            with the shared drop rule (lowest free spot nearest pos[:2]), never inside the pile.
    delete: the topmost box object (same rule as Transfer) goes to the reserve.
    move:   the topmost box object is relocated by the drop rule to the box quadrant centre
            farthest from where it was. object_id/pos None.
    swap:   the topmost box object goes to the reserve and reserve object `object_id` (a
            different shape/colour) takes its place via the drop rule. pos None.
    """

    kind: ViolationKind
    t: float
    object_id: int | None
    pos: Vec3 | None


class Clip(_Strict):
    n_frames: int = Field(gt=0)
    fps: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    render_pass: Literal["flat"] = "flat"

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps

    def frame_times(self) -> list[float]:
        return [i / self.fps for i in range(self.n_frames)]


class Physics(_Strict):
    timestep: float = 1.0 / 240.0
    preroll: float = 1.0
    claw_home: Vec3 = (0.0, 0.0, 0.9)
    claw_travel_z: float = 0.45


class Episode(_Strict):
    spec_version: str = SPEC_VERSION
    episode_id: str
    seed: int
    episode_type: Literal["transfer"]
    split: str
    scene: Scene
    objects: list[ObjectSpec]
    actions: list[Action]
    glides: list[Glide] = []
    violation: Violation | None = None
    twin_of: str | None = None
    clip: Clip
    physics: Physics
    meta: dict

    @model_validator(mode="after")
    def _check(self) -> "Episode":
        ids = [o.id for o in self.objects]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate object ids")
        acts = sorted(self.actions, key=lambda a: a.t_start)
        for a, b in zip(acts, acts[1:]):
            if b.t_start < a.t_end:
                raise ValueError(f"overlapping actions {a} / {b}")
        for a in acts:
            if a.t_end <= a.t_start or a.t_end > self.clip.duration:
                raise ValueError(f"action outside clip or empty: {a}")
        busy = [(a.t_start, a.t_end) for a in acts if isinstance(a, Transfer)]
        screen = [a for a in acts if isinstance(a, ScreenMove)]
        if screen:
            busy.append((screen[0].t_start, max(a.t_end for a in screen)))
        by_obj: dict[int, list[Glide]] = {}
        for g in self.glides:
            if g.t_end <= g.t_start or g.t_end > self.clip.duration:
                raise ValueError(f"glide outside clip or empty: {g}")
            if g.object_id not in ids or next(o for o in self.objects if o.id == g.object_id).start_region == "reserve":
                raise ValueError(f"glide of a missing or reserve object: {g}")
            if any(g.t_start < b and a < g.t_end for a, b in busy):
                raise ValueError(f"glide overlaps a transfer or the screen's raised period: {g}")
            by_obj.setdefault(g.object_id, []).append(g)
        for gs in by_obj.values():
            gs = sorted(gs, key=lambda g: g.t_start)
            if any(b.t_start < a.t_end for a, b in zip(gs, gs[1:])):
                raise ValueError(f"overlapping glides of one object: {gs}")
        if self.violation is not None:
            v = self.violation
            if not self.box_occluded_at(v.t):
                raise ValueError(f"violation at t={v.t} while the box is visible")
            if any(isinstance(a, Transfer) and a.t_start <= v.t <= a.t_end for a in acts):
                raise ValueError("violation during a transfer")
            if v.kind == "add" and (v.object_id is None or v.pos is None):
                raise ValueError("add violation needs object_id and pos")
            if v.kind == "swap" and (v.object_id is None or v.pos is not None):
                raise ValueError("swap violation needs object_id and no pos")
            if v.kind in ("delete", "move") and (v.object_id is not None or v.pos is not None):
                raise ValueError(f"{v.kind} violation takes no object_id/pos")
            if v.object_id is not None:
                o = next((o for o in self.objects if o.id == v.object_id), None)
                if o is None or o.start_region != "reserve":
                    raise ValueError(f"{v.kind} violation must use a reserve object")
        return self

    @property
    def twin_kind(self) -> str:
        """possible | impossible (count changes) | control (count-preserving hidden change)."""
        if self.violation is None:
            return "possible"
        return "impossible" if self.violation.kind in COUNT_CHANGING else "control"

    def screen_intervals(self) -> list[tuple[float, float]]:
        """Intervals where the screen is fully raised."""
        moves = sorted((a for a in self.actions if isinstance(a, ScreenMove)), key=lambda a: a.t_start)
        out, up_since = [], None
        for mv in moves:
            if mv.kind == "screen_raise":
                if up_since is not None:
                    raise ValueError("screen raised twice")
                up_since = mv.t_end
            else:
                if up_since is None:
                    raise ValueError("screen lowered while down")
                out.append((up_since, mv.t_start))
                up_since = None
        if up_since is not None:
            out.append((up_since, self.clip.duration))
        return out

    def box_occluded_at(self, t: float) -> bool:
        return any(a <= t <= b for a, b in self.screen_intervals())

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, s: str) -> "Episode":
        return cls.model_validate(json.loads(s))


def json_schema() -> dict:
    return Episode.model_json_schema()
