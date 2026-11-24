"""Template + seed + split -> concrete Episode. Deterministic in (template, manifest, split, seed)."""

from __future__ import annotations

import colorsys
import math
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from .spec import (
    Camera, Clip, Episode, Glide, Light, ObjectSpec, Physics, Scene, Screen, ScreenMove, Transfer, Tray,
    Violation,
)

SAMPLER_VERSION = "v4"  # v4: optional glide phase
RESERVE_Z = -3.0  # must match backends.base.RESERVE_Z
CLAW_PAD_HALF_Z = 0.006  # must match backends.base.CLAW_PAD_HALF[2]
PILE_H = 0.09     # height above the box floor the screen must hide (validator catches taller piles)
CARRY_MARGIN = 0.03  # carried object's bottom above the camera's sightline over the screen top


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SceneRanges(_Strict):
    """Nuisance variation. Every scene value is drawn from these per episode."""

    cam_azimuth_deg: tuple[float, float] = (-25.0, 25.0)   # 0 = looking along +y
    cam_elevation_deg: tuple[float, float] = (30.0, 48.0)
    cam_distance: tuple[float, float] = (0.85, 1.10)
    cam_fovy_deg: tuple[float, float] = (44.0, 52.0)
    cam_lookat_z: tuple[float, float] = (0.08, 0.20)
    light_elevation_deg: tuple[float, float] = (45.0, 80.0)
    light_intensity: tuple[float, float] = (0.45, 0.75)
    light_tint: float = 0.08
    ambient: tuple[float, float] = (0.25, 0.45)
    box_inner: tuple[float, float] = (0.25, 0.30)
    hopper_inner: tuple[float, float] = (0.19, 0.23)
    box_wall_h: tuple[float, float] = (0.025, 0.04)
    hopper_wall_h: tuple[float, float] = (0.032, 0.04)
    tray_gap: tuple[float, float] = (0.05, 0.10)
    hopper_sides: tuple[Literal["left", "right"], ...] = ("left", "right")
    screen_gap: tuple[float, float] = (0.04, 0.07)        # box outer wall -> screen plane
    palettes: tuple[Literal["random", "single", "two"], ...] = ("random", "single", "two")
    shapes_per_episode: tuple[int, int] = (1, 3)
    min_color_dist: float = 0.25   # RGB distance between any object colour and the tray floors
    # object colour ranges (HSV saturation, value); a generalization probe can move them outside training
    obj_sat: tuple[float, float] = (0.45, 0.9)
    obj_val: tuple[float, float] = (0.6, 0.95)


class Template(_Strict):
    episode_type: Literal["transfer"] = "transfer"
    count_range: tuple[int, int] = (0, 9)        # objects in the box at t = 0
    final_range: tuple[int, int] = (0, 9)        # objects in the box at the end
    k_range: tuple[int, int] = (1, 2)            # objects moved by add/remove
    # "none": the claw only shuffles objects box -> box behind the screen (count unchanged)
    ops: tuple[Literal["add", "remove", "none"], ...] = ("add", "remove", "none")
    op_weights: tuple[float, ...] = (0.4, 0.4, 0.2)
    shuffle_prob: float = 0.35     # add/remove episodes: chance of extra box -> box transfers
    max_transfers: int = 2         # transfers per episode, shuffles included
    count_distribution: Literal["uniform", "zipf"] = "uniform"
    zipf_a: float = 1.5
    shapes: tuple[Literal["sphere", "box", "cylinder"], ...] = ("sphere", "box", "cylinder")
    radius_range: tuple[float, float] = (0.016, 0.03)   # characteristic object radius
    size_jitter: float = 0.12                            # per-object relative jitter
    control_modes: tuple[Literal["size_matched", "area_matched"], ...] = ("size_matched", "area_matched")
    area_ref_count: int = 5      # area_matched: total area equals this many objects at mid radius
    use_screen: bool = True
    # height above the box floor the screen hides; None = PILE_H (v2/v3). Large objects need more.
    pile_h: float | None = None
    # occluded: every transfer behind the raised screen (the twins' setting, v2)
    # visible:  the screen never rises; every transfer in view
    # mixed:    the first transfer in view, then the screen rises for the rest (needs >= 2 transfers;
    #           episodes with one transfer draw from the other modes)
    visibility_modes: tuple[Literal["occluded", "visible", "mixed"], ...] = ("occluded",)
    visibility_weights: tuple[float, ...] = (1.0,)
    # glide phase (v5, stage 0 individuation): before the first action, objects in view move on their own
    # to random points in their tray (common fate), several at once. (0, 0) = none (no rng draws).
    glide_phase_s: tuple[float, float] = (0.0, 0.0)
    glide_prob: float = 0.8           # per object in the box or hopper
    glide_s: tuple[float, float] = (0.7, 1.5)
    transfer_s: tuple[float, float] = (1.6, 2.0)
    screen_move_s: tuple[float, float] = (0.3, 0.5)
    clip: Clip = Clip(n_frames=32, fps=4, width=256, height=256)
    physics: Physics = Physics()
    scene: SceneRanges = SceneRanges()

    @model_validator(mode="after")
    def _fits(self) -> "Template":
        if len(self.op_weights) != len(self.ops):
            raise ValueError("op_weights must match ops")
        if len(self.visibility_weights) != len(self.visibility_modes):
            raise ValueError("visibility_weights must match visibility_modes")
        if not self.use_screen and self.visibility_modes != ("visible",):
            raise ValueError("use_screen=False needs visibility_modes=('visible',)")
        if self.visibility_modes == ("mixed",):
            raise ValueError("mixed alone cannot serve one-transfer episodes; add occluded or visible")
        if self.k_range[1] > self.max_transfers:
            raise ValueError("k_range exceeds max_transfers")
        sm = self.screen_move_s[1]
        if self.glide_phase_s[0] > self.glide_phase_s[1] or (self.glide_phase_s[1] > 0 and self.glide_s[0] > self.glide_phase_s[0]):
            raise ValueError("glide_phase_s must be ordered and at least as long as the shortest glide")
        worst = self.glide_phase_s[1] + 0.9 + sm + 0.4 + self.max_transfers * (self.transfer_s[1] + 0.1) + 0.7 + sm + 0.6
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
    w = np.asarray(t.op_weights, dtype=float)
    for _ in range(10_000):
        n0 = int(rng.choice(vals, p=p))
        op = str(rng.choice(list(t.ops), p=w / w.sum()))
        k = 0 if op == "none" else int(rng.integers(t.k_range[0], t.k_range[1] + 1))
        if op == "none" and n0 == 0:
            continue  # nothing to shuffle
        final = n0 + k if op == "add" else n0 - k
        if not (t.final_range[0] <= final <= t.final_range[1]) or final < 0:
            continue
        shapes = shapes_for(n0 + (k if op == "add" else 0) + 1)
        if _split_ok(split, n0, op, k, final, shapes, man):
            return n0, op, k, shapes
    raise RuntimeError(f"no valid (count, op, k) for split {split!r} under this template/manifest")


def _hsv(rng, s, v, h=None) -> tuple[float, float, float, float]:
    r, g, b = colorsys.hsv_to_rgb(rng.uniform(0, 1) if h is None else h, rng.uniform(*s), rng.uniform(*v))
    return (float(r), float(g), float(b), 1.0)


def _cdist(a, b) -> float:
    return float(np.linalg.norm(np.subtract(a[:3], b[:3])))


def _volume(shape: str, size: tuple[float, ...]) -> float:
    if shape == "sphere":
        return 4 / 3 * math.pi * size[0] ** 3
    if shape == "box":
        return 8 * size[0] * size[1] * size[2]
    return 2 * math.pi * size[0] ** 2 * size[1]


def _project(cam: Camera, aspect: float, pts: np.ndarray) -> np.ndarray:
    """World points [N, 3] -> NDC xy [N, 2] (|v| <= 1 is on screen). Raises if a point is behind the camera."""
    pos, look = np.asarray(cam.pos), np.asarray(cam.lookat)
    f = (look - pos) / np.linalg.norm(look - pos)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = pts - pos
    z = d @ f
    if np.any(z <= 0.01):
        raise ValueError("point behind camera")
    th = math.tan(math.radians(cam.fovy) / 2)
    return np.stack([(d @ r) / (z * th * aspect), (d @ u) / (z * th)], axis=1)


def _tray_corners(tr: Tray, top: float) -> np.ndarray:
    hx = tr.inner_size[0] / 2 + tr.wall_thickness
    hy = tr.inner_size[1] / 2 + tr.wall_thickness
    cx, cy = tr.center
    return np.array([[cx + sx * hx, cy + sy * hy, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, top)])


def _fit_screen(cam: Camera, box: Tray, gap: float, pile_h: float = PILE_H,
                margin: float = 0.02) -> tuple[float, float, float, float]:
    """Screen (center_x, y, yaw, width, height): a vertical panel facing the camera's horizontal
    direction, `gap` in front of the box's outer wall, sized so that every point of the box interior
    up to pile_h lies behind it from the camera."""
    c = np.asarray(cam.pos)
    b = np.asarray(box.center)
    d = c[:2] - b
    n = d / np.linalg.norm(d)                 # panel normal, toward the camera
    yaw = math.atan2(n[0], -n[1])
    t = np.array([math.cos(yaw), math.sin(yaw)])  # panel tangent
    hx, hy = box.inner_size[0] / 2 + box.wall_thickness, box.inner_size[1] / 2 + box.wall_thickness
    p0 = b + n * (abs(n[0]) * hx + abs(n[1]) * hy + gap)
    ss, zs = [], []
    ix, iy = box.inner_size[0] / 2, box.inner_size[1] / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            for z in (box.wall_thickness, box.wall_thickness + pile_h):
                pt = np.array([b[0] + sx * ix, b[1] + sy * iy, z])
                lam = float((p0 - c[:2]) @ n) / float((pt[:2] - c[:2]) @ n)
                q = c + lam * (pt - c)
                ss.append(float((q[:2] - p0) @ t))
                zs.append(float(q[2]))
    s_c = (min(ss) + max(ss)) / 2
    width = max(ss) - min(ss) + 2 * margin
    height = max(zs) + margin
    cxy = p0 + s_c * t
    return float(cxy[0]), float(cxy[1]), float(yaw), float(width), float(height)


def _screen_clear_of(screen: Screen, tr: Tray, margin: float = 0.02) -> bool:
    t = np.array([math.cos(screen.yaw), math.sin(screen.yaw)])
    hx = tr.inner_size[0] / 2 + tr.wall_thickness + margin
    hy = tr.inner_size[1] / 2 + tr.wall_thickness + margin
    for u in np.linspace(-screen.width / 2, screen.width / 2, 25):
        x, y = screen.center_x + u * t[0], screen.y + u * t[1]
        if abs(x - tr.center[0]) < hx and abs(y - tr.center[1]) < hy:
            return False
    return True


def _carry_height(cam: Camera, box: Tray, screen: Screen, obj_h: float) -> float:
    """Claw travel height at which a carried object (height obj_h under the claw pad) clears the
    camera's sightline over the screen top everywhere above the box interior."""
    c = np.asarray(cam.pos)
    n = np.array([math.sin(screen.yaw), -math.cos(screen.yaw)])     # screen normal, toward camera
    d_s = float((c[:2] - np.array([screen.center_x, screen.y])) @ n)  # camera -> screen plane
    ix, iy = box.inner_size[0] / 2, box.inner_size[1] / 2
    d_near = float((c[:2] - np.asarray(box.center)) @ n) - (abs(n[0]) * ix + abs(n[1]) * iy)
    if d_near <= d_s:
        raise ValueError("box interior is not behind the screen")
    z_line = c[2] - (c[2] - screen.height) * d_near / d_s   # sightline height above the box's near edge
    return float(z_line + obj_h + CARRY_MARGIN)


def _sample_scene(rng, t: Template) -> tuple[Scene, Physics, dict]:
    R = t.scene
    for _attempt in range(200):
        box_in = (float(rng.uniform(*R.box_inner)), float(rng.uniform(*R.box_inner)))
        hop_in = (float(rng.uniform(*R.hopper_inner)), float(rng.uniform(*R.hopper_inner)))
        side = str(rng.choice(list(R.hopper_sides)))
        wt = 0.008
        sep = box_in[0] / 2 + hop_in[0] / 2 + 2 * wt + float(rng.uniform(*R.tray_gap))
        mid = float(rng.uniform(-0.08, -0.03))
        sgn = -1.0 if side == "left" else 1.0
        floor = _hsv(rng, (0.0, 0.4), (0.15, 0.55))
        box = Tray(name="box", center=(mid - sgn * sep / 2, float(rng.uniform(-0.01, 0.01))), inner_size=box_in,
                   wall_height=float(rng.uniform(*R.box_wall_h)), wall_thickness=wt, color=floor)
        hopper = Tray(name="hopper", center=(mid + sgn * sep / 2, float(rng.uniform(-0.01, 0.01))), inner_size=hop_in,
                      wall_height=float(rng.uniform(*R.hopper_wall_h)), wall_thickness=wt,
                      color=_hsv(rng, (0.0, 0.4), (0.15, 0.55)))

        az = math.radians(rng.uniform(*R.cam_azimuth_deg))
        el = math.radians(rng.uniform(*R.cam_elevation_deg))
        dist = float(rng.uniform(*R.cam_distance))
        look = np.array([mid + rng.uniform(-0.02, 0.02), rng.uniform(-0.03, 0.02), rng.uniform(*R.cam_lookat_z)])
        pos = look + dist * np.array([math.cos(el) * math.sin(az), -math.cos(el) * math.cos(az), math.sin(el)])
        cam = Camera(pos=tuple(float(x) for x in pos), lookat=tuple(float(x) for x in look),
                     fovy=float(rng.uniform(*R.cam_fovy_deg)))

        sx, sy, yaw, sw, shh = _fit_screen(cam, box, float(rng.uniform(*R.screen_gap)),
                                           PILE_H if t.pile_h is None else t.pile_h)
        screen = Screen(center_x=sx, y=sy, yaw=yaw, width=sw, height=shh, thickness=0.01,
                        color=_hsv(rng, (0.0, 0.4), (0.4, 0.9)))
        if not _screen_clear_of(screen, hopper):
            continue
        # carried object height: largest object (2 * max radius incl. jitter) plus the claw pad
        obj_h = 2 * t.radius_range[1] * (1 + t.size_jitter) + 2 * CLAW_PAD_HALF_Z
        travel_z = _carry_height(cam, box, screen, obj_h)
        aspect = t.clip.width / t.clip.height
        # where the claw can be while carrying: above the tray interiors (object centres sit >= r inside)
        carry = np.array([[tr.center[0] + sx * (tr.inner_size[0] / 2 - 0.02), tr.center[1] + sy * (tr.inner_size[1] / 2 - 0.02), 0.0]
                          for tr in (box, hopper) for sx in (-1, 1) for sy in (-1, 1)])
        carry_lo = carry + [0.0, 0.0, travel_z - obj_h]
        carry_hi = carry + [0.0, 0.0, travel_z + 0.01]
        pts = np.concatenate([_tray_corners(box, box.wall_height), _tray_corners(hopper, hopper.wall_height),
                              carry_lo, carry_hi])
        if np.abs(_project(cam, aspect, pts)).max() > 0.95:
            continue  # a tray or the carry path would leave the frame

        laz = rng.uniform(0, 2 * math.pi)
        lel = math.radians(rng.uniform(*R.light_elevation_deg))
        ldir = (-math.cos(lel) * math.cos(laz), -math.cos(lel) * math.sin(laz), -math.sin(lel))
        inten = float(rng.uniform(*R.light_intensity))
        diffuse = tuple(float(inten * (1 + rng.uniform(-R.light_tint, R.light_tint))) for _ in range(3))
        lights = [Light(pos=tuple(float(-3 * x) for x in ldir), dir=tuple(float(x) for x in ldir), diffuse=diffuse)]
        amb = float(rng.uniform(*R.ambient))
        bg = _hsv(rng, (0.0, 0.35), (0.3, 0.85))
        scene = Scene(table_size=(0.8, 0.6), table_color=_hsv(rng, (0.0, 0.3), (0.45, 0.85)), background=bg[:3],
                      ambient=(amb, amb, amb), box=box, hopper=hopper, screen=screen, camera=cam, lights=lights)
        phys = t.physics.model_copy(update={"claw_home": (mid, 0.0, t.physics.claw_home[2]),
                                             "claw_travel_z": travel_z})
        params = {"cam_azimuth_deg": math.degrees(az), "cam_elevation_deg": math.degrees(el), "cam_distance": dist,
                  "hopper_side": side, "light_azimuth_deg": math.degrees(laz), "light_elevation_deg": math.degrees(lel)}
        return scene, phys, params
    raise RuntimeError("could not sample a scene that fits the frame in 200 attempts; ranges too wide")


def _object_colors(rng, R: SceneRanges, n: int, floors: list) -> tuple[str, list]:
    palette = str(rng.choice(list(R.palettes)))

    def draw():
        for _ in range(200):
            c = _hsv(rng, R.obj_sat, R.obj_val)
            if all(_cdist(c, f) >= R.min_color_dist for f in floors):
                return c
        raise RuntimeError("no object colour far enough from the tray floors")

    if palette == "random":
        return palette, [draw() for _ in range(n)]
    pool = [draw()] if palette == "single" else [draw(), draw()]
    return palette, [pool[int(rng.integers(len(pool)))] for _ in range(n)]


def sample_episode(t: Template, man: Manifest, split: str, seed: int) -> Episode:
    rng = np.random.default_rng(seed)
    R = t.scene
    lo_s, hi_s = R.shapes_per_episode
    ep_shapes = [str(x) for x in rng.choice(list(t.shapes), size=min(int(rng.integers(lo_s, hi_s + 1)), len(t.shapes)),
                                            replace=False)]

    def shapes_for(n: int) -> list[str]:
        return [str(x) for x in rng.choice(ep_shapes, size=n)]

    n0, op, k, shapes = _sample_counts(rng, t, man, split, shapes_for)
    final = n0 + k if op == "add" else n0 - k

    scene, physics, scene_params = _sample_scene(rng, t)
    box, hopper = scene.box, scene.hopper

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
    shapes = shapes[:n_total]
    radii = [float(base_r * rng.uniform(1 - t.size_jitter, 1 + t.size_jitter)) for _ in range(n_total)]
    descr = [(shapes[i], _size_for(shapes[i], radii[i]), radii[i]) for i in range(n_total)]
    palette, colors = _object_colors(rng, R, n_total, [box.color, hopper.color])

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
        objects.append(ObjectSpec(id=i, shape=shape, size=size, color=colors[i], mass=float(_volume(shape, size) * 600),
                                  start_region=region, start_pos=pos, start_quat=quat))

    # --- transfers: k add/remove moves plus box -> box shuffles, in random order ---
    if op == "none":
        n_shuf = int(rng.integers(1, t.max_transfers + 1))
    elif k < t.max_transfers and rng.uniform() < t.shuffle_prob:
        n_shuf = int(rng.integers(1, t.max_transfers - k + 1))
    else:
        n_shuf = 0
    moves = (["op"] * k) + (["shuffle"] * n_shuf)
    for _ in range(100):
        order = [moves[i] for i in rng.permutation(len(moves))]
        in_box, ok = n0, True
        for m_ in order:  # a shuffle needs something in the box at that moment
            if m_ == "shuffle" and in_box == 0:
                ok = False
                break
            if m_ == "op":
                in_box += 1 if op == "add" else -1
        if ok:
            break
    else:
        raise RuntimeError("no transfer order keeps the box non-empty for every shuffle")

    def drop_xy(tray: Tray) -> tuple[float, float]:
        m = 0.045
        return (float(tray.center[0] + rng.uniform(-tray.inner_size[0] / 2 + m, tray.inner_size[0] / 2 - m)),
                float(tray.center[1] + rng.uniform(-tray.inner_size[1] / 2 + m, tray.inner_size[1] / 2 - m)))

    if t.visibility_modes == ("occluded",):
        vis = "occluded"   # no draw: v2 episodes stay identical for the same seed
    else:
        modes = [(m, w) for m, w in zip(t.visibility_modes, t.visibility_weights) if m != "mixed" or len(order) >= 2]
        w = np.array([w for _, w in modes], dtype=float)
        if w.sum() <= 0:
            raise ValueError(f"no visibility mode with weight > 0 for a {len(order)}-transfer episode: "
                             f"{dict(zip(t.visibility_modes, t.visibility_weights))}")
        vis = str(rng.choice([m for m, _ in modes], p=w / w.sum()))

    def raise_screen(tt: float) -> float:
        sm = float(rng.uniform(*t.screen_move_s))
        actions.append(ScreenMove(kind="screen_raise", t_start=tt, t_end=tt + sm))
        return tt + sm + float(rng.uniform(0.1, 0.4))

    actions = []
    g_phase = float(rng.uniform(*t.glide_phase_s)) if t.glide_phase_s[1] > 0 else 0.0
    tt = g_phase + float(rng.uniform(0.4, 0.9))
    if vis == "occluded":
        tt = raise_screen(tt)
    for i_m, m_ in enumerate(order):
        if vis == "mixed" and i_m == 1:
            tt = raise_screen(tt)
        if m_ == "op":
            src, dst = ("hopper", "box") if op == "add" else ("box", "hopper")
        else:
            src, dst = "box", "box"
        d = float(rng.uniform(*t.transfer_s))
        actions.append(Transfer(src=src, dst=dst, t_start=tt, t_end=tt + d, drop_xy=drop_xy(box if dst == "box" else hopper)))
        tt += d + 0.1
    last_transfer_end = tt - 0.1
    if vis != "visible":
        tt += float(rng.uniform(0.4, 0.7))
        sm = float(rng.uniform(*t.screen_move_s))
        actions.append(ScreenMove(kind="screen_lower", t_start=tt, t_end=tt + sm))
        tt += sm
    if tt + 0.6 > t.clip.duration:
        raise RuntimeError(f"timeline {tt:.2f}s overruns clip {t.clip.duration}s")

    glides = []
    if g_phase > 0:
        movers = [o for o in objects if o.start_region != "reserve"]
        chosen = [o for o in movers if rng.uniform() < t.glide_prob]
        if movers and not chosen:
            chosen = [movers[int(rng.integers(len(movers)))]]
        for o in chosen:
            tray = box if o.start_region == "box" else hopper
            d = min(float(rng.uniform(*t.glide_s)), g_phase)
            t0 = float(rng.uniform(0.2, 0.2 + g_phase - d))
            m = radii[o.id] + 0.012
            hx, hy = tray.inner_size[0] / 2 - m, tray.inner_size[1] / 2 - m
            # a target clear of every other object's start spot and of the targets already chosen
            others = [(p.start_pos[0], p.start_pos[1], radii[p.id]) for p in movers if p.id != o.id]
            others += [(g.to_xy[0], g.to_xy[1], radii[g.object_id]) for g in glides]
            for _ in range(200):
                to = (float(tray.center[0] + rng.uniform(-hx, hx)), float(tray.center[1] + rng.uniform(-hy, hy)))
                if all(math.hypot(to[0] - x, to[1] - y) > radii[o.id] + r + 0.015 for x, y, r in others):
                    break
            else:
                continue  # crowded tray: this object stays put
            glides.append(Glide(object_id=o.id, t_start=t0, t_end=t0 + d, to_xy=to))

    return Episode(
        episode_id=f"{split}-{seed:09d}", seed=seed, episode_type="transfer", split=split, scene=scene,
        objects=objects, actions=actions, glides=glides, clip=t.clip, physics=physics,
        meta={"sampler": SAMPLER_VERSION, "n0": n0, "op": op, "k": k, "final": final, "shuffles": n_shuf,
              "visibility": vis, "control_mode": mode, "base_radius": base_r, "palette": palette, "shapes": ep_shapes,
              "n_hopper": n_hopper, "last_transfer_end": last_transfer_end,
              "glide_phase": g_phase, "n_glides": len(glides), "scene": scene_params},
    )


def can_control(ep: Episode) -> bool:
    """Control twins (move/swap) need something in the box after the last transfer."""
    return ep.meta["final"] > 0


def make_twin(ep: Episode, seed: int, kind: Literal["impossible", "control"] = "impossible") -> Episode:
    """Twin of a possible screened episode, identical until a hidden change while the box is hidden.

    impossible: add or delete one object (the count shown at the reveal is off by one).
    control:    move or swap one object (count unchanged; an unexplained change all the same).
    """
    if ep.violation is not None:
        raise ValueError("already a twin")
    if ep.meta["visibility"] != "occluded":
        raise ValueError(f"twins are made only of occluded episodes, not {ep.meta['visibility']}")
    rng = np.random.default_rng([seed, 7 if kind == "impossible" else 11])
    lower = next(a for a in ep.actions if isinstance(a, ScreenMove) and a.kind == "screen_lower")
    lo, hi = ep.meta["last_transfer_end"] + 0.15, lower.t_start - 0.15
    if hi <= lo:
        raise RuntimeError("no occluded window for a violation")
    t_v = float(rng.uniform(lo, hi))
    objects = list(ep.objects)
    res = next(o for o in ep.objects if o.start_region == "reserve")
    meta = dict(ep.meta)
    if kind == "impossible":
        vk = str(rng.choice(["add"] + (["delete"] if ep.meta["final"] > 0 else [])))
        if vk == "add":
            b = ep.scene.box
            m = 0.045
            pos = (float(b.center[0] + rng.uniform(-b.inner_size[0] / 2 + m, b.inner_size[0] / 2 - m)),
                   float(b.center[1] + rng.uniform(-b.inner_size[1] / 2 + m, b.inner_size[1] / 2 - m)), 0.07)
            v = Violation(kind="add", t=t_v, object_id=res.id, pos=pos)
        else:
            v = Violation(kind="delete", t=t_v, object_id=None, pos=None)
        meta["shown_final"] = ep.meta["final"] + (1 if vk == "add" else -1)
    elif kind == "control":
        if not can_control(ep):
            raise ValueError("control twin needs a non-empty box at the end")
        # A swap restyles the reserve with a colour far from every visible object's. The reserve is
        # never visible before the violation, so the twins stay pixel-identical up to t_v. A crowded
        # random palette can leave no such colour; then a swap is not a valid control for this scene
        # and only move is offered (recorded in meta["swap_feasible"]).
        visible = [o.color for o in ep.objects if o.start_region != "reserve"]
        floors = [ep.scene.box.color, ep.scene.hopper.color]
        c = None
        for _ in range(500):
            cand = _hsv(rng, (0.45, 0.9), (0.6, 0.95))
            if all(_cdist(cand, f) >= 0.25 for f in floors) and all(_cdist(cand, o) >= 0.35 for o in visible):
                c = cand
                break
        meta["swap_feasible"] = c is not None
        vk = str(rng.choice(["move", "swap"])) if c is not None else "move"
        if vk == "move":
            v = Violation(kind="move", t=t_v, object_id=None, pos=None)
        else:
            shape = str(rng.choice(["sphere", "box", "cylinder"]))
            r = float(ep.meta["base_radius"])
            size = _size_for(shape, r)
            new = res.model_copy(update={"shape": shape, "size": size, "color": c, "mass": float(_volume(shape, size) * 600),
                                         "start_pos": (res.start_pos[0], 0.0, RESERVE_Z + _z_extent(shape, size) + 0.002)})
            objects = [new if o.id == res.id else o for o in ep.objects]
            v = Violation(kind="swap", t=t_v, object_id=res.id, pos=None)
        meta["shown_final"] = ep.meta["final"]
    else:
        raise ValueError(f"unknown twin kind {kind!r}")
    suffix = "-imp" if kind == "impossible" else "-ctl"
    # re-validate (model_copy would skip the occlusion checks)
    return Episode.model_validate({**ep.model_dump(), "episode_id": ep.episode_id + suffix,
                                   "objects": [o.model_dump() for o in objects],
                                   "violation": v.model_dump(), "twin_of": ep.episode_id, "meta": meta})
