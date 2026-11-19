"""Template + seed + split -> concrete Episode. Deterministic in (template, manifest, split, seed)."""

from __future__ import annotations

import colorsys
import math
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from .spec import (
    Camera, Clip, Episode, Light, ObjectSpec, Physics, Scene, Screen, ScreenMove, Transfer, Tray,
    Violation,
)

SCREEN_MOVE_S = 0.4
RESERVE_Z = -3.0  # must match backends.base.RESERVE_Z


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Template(_Strict):
    episode_type: Literal["transfer"] = "transfer"
    count_range: tuple[int, int] = (0, 9)        # objects in the box at t = 0
    final_range: tuple[int, int] = (0, 9)        # objects in the box at the end
    k_range: tuple[int, int] = (1, 2)            # objects moved
    ops: tuple[Literal["add", "remove"], ...] = ("add", "remove")
    count_distribution: Literal["uniform", "zipf"] = "uniform"
    zipf_a: float = 1.5
    shapes: tuple[Literal["sphere", "box", "cylinder"], ...] = ("sphere", "box", "cylinder")
    radius_range: tuple[float, float] = (0.016, 0.03)   # characteristic object radius
    size_jitter: float = 0.12                            # per-object relative jitter
    control_modes: tuple[Literal["size_matched", "area_matched"], ...] = ("size_matched", "area_matched")
    area_ref_count: int = 5      # area_matched: total area equals this many objects at mid radius
    use_screen: bool = True
    transfer_s: tuple[float, float] = (1.6, 2.0)
    clip: Clip = Clip(n_frames=32, fps=4, width=128, height=128)
    physics: Physics = Physics()

    @model_validator(mode="after")
    def _fits(self) -> "Template":
        worst = 0.9 + SCREEN_MOVE_S + 0.4 + self.k_range[1] * (self.transfer_s[1] + 0.1) + 0.7 + SCREEN_MOVE_S + 0.6
        if self.use_screen and worst > self.clip.duration:
            raise ValueError(f"clip of {self.clip.duration}s cannot hold worst-case timeline {worst:.2f}s")
        return self


class Manifest(_Strict):
    """Held-out splits, fixed at generation time."""

    heldout_counts: tuple[int, ...] = (10, 11, 12, 13, 14, 15)
    # (initial count, op, k) combinations never seen in training, e.g. 4 + 3
    heldout_combos: tuple[tuple[int, Literal["add", "remove"], int], ...] = ((4, "add", 2), (2, "add", 2), (7, "remove", 2))
    heldout_shapes: tuple[str, ...] = ()


def load_yaml(path: str, cls):
    with open(path) as f:
        return cls.model_validate(yaml.safe_load(f) or {})


def _split_ok(split: str, n0: int, op: str, k: int, final: int, shapes: list[str], man: Manifest) -> bool:
    in_counts = n0 in man.heldout_counts or final in man.heldout_counts
    in_combo = (n0, op, k) in man.heldout_combos
    in_shape = any(s in man.heldout_shapes for s in shapes)
    if split == "train":
        return not (in_counts or in_combo or in_shape)
    if split == "test_counts":
        return in_counts
    if split == "test_combos":
        return in_combo and not in_counts
    if split == "test_shapes":
        return in_shape and not in_counts
    raise ValueError(f"unknown split {split!r}")


def _rand_color(rng, s=(0.45, 0.9), v=(0.6, 0.95)) -> tuple[float, float, float, float]:
    r, g, b = colorsys.hsv_to_rgb(rng.uniform(0, 1), rng.uniform(*s), rng.uniform(*v))
    return (r, g, b, 1.0)


def _muted(rng, lo, hi) -> tuple[float, float, float, float]:
    base = rng.uniform(lo, hi)
    tint = rng.uniform(-0.05, 0.05, 3)
    r, g, b = np.clip(base + tint, 0, 1)
    return (float(r), float(g), float(b), 1.0)


def _size_for(shape: str, r: float) -> tuple[float, ...]:
    if shape == "sphere":
        return (r,)
    if shape == "box":
        h = r * 0.8  # half-extent so the cube's area matches a sphere of radius r roughly
        return (h, h, h)
    return (r, r * 0.8)


def _z_extent(shape: str, size: tuple[float, ...]) -> float:
    if shape == "sphere":
        return size[0]
    if shape == "box":
        return size[2]
    return size[1]


def _place_in(rng, tray: Tray, objs: list[tuple[str, tuple[float, ...], float]]) -> list[tuple[float, float, float]]:
    """Non-overlapping xy positions for objects (shape, size, radius) in a tray; stacks if crowded."""
    hx, hy = tray.inner_size[0] / 2, tray.inner_size[1] / 2
    placed: list[tuple[float, float, float, float]] = []  # x, y, z, r
    out = []
    for shape, size, r in objs:
        layer = 0
        while True:
            z = tray.wall_thickness + _z_extent(shape, size) + 0.004 + layer * 0.07
            for _ in range(200):
                x = tray.center[0] + rng.uniform(-hx + r + 0.004, hx - r - 0.004)
                y = tray.center[1] + rng.uniform(-hy + r + 0.004, hy - r - 0.004)
                if all((x - px) ** 2 + (y - py) ** 2 > (r + pr + 0.004) ** 2 or abs(z - pz) > 0.06
                       for px, py, pz, pr in placed):
                    placed.append((x, y, z, r))
                    out.append((float(x), float(y), float(z)))
                    break
            else:
                layer += 1
                continue
            break
    return out


def _sample_counts(rng, t: Template, man: Manifest, split: str, shapes_for) -> tuple[int, str, int, list[str]]:
    lo, hi = t.count_range
    vals = np.arange(lo, hi + 1)
    if t.count_distribution == "uniform":
        p = np.ones(len(vals))
    else:
        p = 1.0 / np.power(vals - lo + 1, t.zipf_a)
    p = p / p.sum()
    for _ in range(10_000):
        n0 = int(rng.choice(vals, p=p))
        op = str(rng.choice(list(t.ops)))
        k = int(rng.integers(t.k_range[0], t.k_range[1] + 1))
        final = n0 + k if op == "add" else n0 - k
        if not (t.final_range[0] <= final <= t.final_range[1]) or final < 0:
            continue
        shapes = shapes_for(n0 + (k if op == "add" else 0) + 1)
        if _split_ok(split, n0, op, k, final, shapes, man):
            return n0, op, k, shapes
    raise RuntimeError(f"no valid (count, op, k) for split {split!r} under this template/manifest")


def sample_episode(t: Template, man: Manifest, split: str, seed: int) -> Episode:
    rng = np.random.default_rng(seed)

    def shapes_for(n: int) -> list[str]:
        return [str(s) for s in rng.choice(list(t.shapes), size=n)]

    n0, op, k, shapes = _sample_counts(rng, t, man, split, shapes_for)
    final = n0 + k if op == "add" else n0 - k

    # --- scene ---
    tray_floor = _muted(rng, 0.15, 0.45)
    box = Tray(name="box", center=(float(0.1 + rng.uniform(-0.015, 0.015)), 0.0), inner_size=(0.28, 0.28),
               wall_height=0.03, wall_thickness=0.008, color=tray_floor)
    hopper = Tray(name="hopper", center=(float(-0.225 + rng.uniform(-0.01, 0.01)), 0.0), inner_size=(0.21, 0.21),
                  wall_height=0.035, wall_thickness=0.008, color=_muted(rng, 0.15, 0.45))
    screen = Screen(center_x=box.center[0], y=-0.2, width=0.36, height=0.38, thickness=0.01,
                    color=_muted(rng, 0.5, 0.85))
    cam = Camera(pos=(float(-0.04 + rng.uniform(-0.02, 0.02)), float(-0.55 + rng.uniform(-0.02, 0.02)),
                      float(0.6 + rng.uniform(-0.02, 0.02))),
                 lookat=(float(-0.055 + rng.uniform(-0.015, 0.015)), float(-0.01 + rng.uniform(-0.015, 0.015)), 0.02),
                 fovy=float(rng.uniform(43, 45)))
    lights = [Light(pos=(float(rng.uniform(-0.6, 0.6)), float(rng.uniform(-0.8, -0.2)), 1.6),
                    dir=(0.0, 0.3, -1.0), diffuse=tuple(float(x) for x in np.full(3, rng.uniform(0.45, 0.7))))]
    bg = float(rng.uniform(0.35, 0.8))
    scene = Scene(table_size=(0.8, 0.6), table_color=_muted(rng, 0.55, 0.8),
                  background=(bg, bg, float(min(1.0, bg + rng.uniform(-0.05, 0.08)))),
                  ambient=tuple(float(x) for x in np.full(3, rng.uniform(0.3, 0.45))),
                  box=box, hopper=hopper, screen=screen, camera=cam, lights=lights)

    # --- objects (shortcut controls: per-episode base radius set by control mode) ---
    mode = str(rng.choice(list(t.control_modes)))
    rlo, rhi = t.radius_range
    if mode == "size_matched":
        base_r = rng.uniform(rlo, rhi)
    else:
        mid = (rlo + rhi) / 2
        base_r = mid * math.sqrt(t.area_ref_count / max(final, 1))
        base_r = float(np.clip(base_r, rlo, rhi))
    n_hopper = (k + int(rng.integers(0, 3))) if op == "add" else int(rng.integers(0, 4))
    n_total = n0 + n_hopper + 1  # +1 reserve
    if len(shapes) < n_total:
        shapes = shapes + shapes_for(n_total - len(shapes))
    radii = [float(base_r * rng.uniform(1 - t.size_jitter, 1 + t.size_jitter)) for _ in range(n_total)]
    descr = [(shapes[i], _size_for(shapes[i], radii[i]), radii[i]) for i in range(n_total)]

    box_pos = _place_in(rng, box, descr[:n0])
    hop_pos = _place_in(rng, hopper, descr[n0:n0 + n_hopper])
    objects = []
    for i, (shape, size, r) in enumerate(descr):
        if i < n0:
            region, pos = "box", box_pos[i]
        elif i < n0 + n_hopper:
            region, pos = "hopper", hop_pos[i - n0]
        else:
            region, pos = "reserve", (i * 0.3, 0.0, RESERVE_Z + _z_extent(shape, size) + 0.002)
        ang = rng.uniform(0, 2 * math.pi)
        quat = (float(math.cos(ang / 2)), 0.0, 0.0, float(math.sin(ang / 2)))
        if shape == "sphere":
            vol = 4 / 3 * math.pi * size[0] ** 3
        elif shape == "box":
            vol = 8 * size[0] * size[1] * size[2]
        else:
            vol = 2 * math.pi * size[0] ** 2 * size[1]
        objects.append(ObjectSpec(id=i, shape=shape, size=size, color=_rand_color(rng), mass=float(vol * 600),
                                  start_region=region, start_pos=pos, start_quat=quat))

    # --- timeline ---
    def drop_xy(tray: Tray) -> tuple[float, float]:
        m = 0.045
        return (float(tray.center[0] + rng.uniform(-tray.inner_size[0] / 2 + m, tray.inner_size[0] / 2 - m)),
                float(tray.center[1] + rng.uniform(-tray.inner_size[1] / 2 + m, tray.inner_size[1] / 2 - m)))

    actions = []
    tt = float(rng.uniform(0.4, 0.9))
    if t.use_screen:
        actions.append(ScreenMove(kind="screen_raise", t_start=tt, t_end=tt + SCREEN_MOVE_S))
        tt += SCREEN_MOVE_S + float(rng.uniform(0.1, 0.4))
    src, dst = ("hopper", "box") if op == "add" else ("box", "hopper")
    for _ in range(k):
        d = float(rng.uniform(*t.transfer_s))
        actions.append(Transfer(src=src, dst=dst, t_start=tt, t_end=tt + d, drop_xy=drop_xy(box if dst == "box" else hopper)))
        tt += d + 0.1
    last_transfer_end = tt - 0.1
    if t.use_screen:
        tt += float(rng.uniform(0.4, 0.7))
        actions.append(ScreenMove(kind="screen_lower", t_start=tt, t_end=tt + SCREEN_MOVE_S))
        tt += SCREEN_MOVE_S
    if tt + 0.6 > t.clip.duration:
        raise RuntimeError(f"timeline {tt:.2f}s overruns clip {t.clip.duration}s")

    return Episode(
        episode_id=f"{split}-{seed:09d}", seed=seed, episode_type="transfer", split=split, scene=scene,
        objects=objects, actions=actions, clip=t.clip, physics=t.physics,
        meta={"n0": n0, "op": op, "k": k, "final": final, "control_mode": mode, "base_radius": base_r,
              "n_hopper": n_hopper, "last_transfer_end": last_transfer_end},
    )


def make_twin(ep: Episode, seed: int) -> Episode:
    """Impossible twin of a possible screened episode: identical until a hidden add/delete."""
    if ep.violation is not None:
        raise ValueError("already a twin")
    rng = np.random.default_rng([seed, 7])
    lower = next(a for a in ep.actions if isinstance(a, ScreenMove) and a.kind == "screen_lower")
    lo, hi = ep.meta["last_transfer_end"] + 0.15, lower.t_start - 0.15
    if hi <= lo:
        raise RuntimeError("no occluded window for a violation")
    kinds = ["add"] + (["delete"] if ep.meta["final"] > 0 else [])
    kind = str(rng.choice(kinds))
    t_v = float(rng.uniform(lo, hi))
    if kind == "add":
        res = next(o for o in ep.objects if o.start_region == "reserve")
        b = ep.scene.box
        m = 0.045
        pos = (float(b.center[0] + rng.uniform(-b.inner_size[0] / 2 + m, b.inner_size[0] / 2 - m)),
               float(b.center[1] + rng.uniform(-b.inner_size[1] / 2 + m, b.inner_size[1] / 2 - m)), 0.07)
        v = Violation(kind="add", t=t_v, object_id=res.id, pos=pos)
    else:
        v = Violation(kind="delete", t=t_v, object_id=None, pos=None)
    meta = dict(ep.meta)
    meta["shown_final"] = ep.meta["final"] + (1 if kind == "add" else -1)
    # re-validate (model_copy would skip the occlusion checks)
    return Episode.model_validate({**ep.model_dump(), "episode_id": ep.episode_id + "-imp",
                                   "violation": v.model_dump(), "twin_of": ep.episode_id, "meta": meta})
