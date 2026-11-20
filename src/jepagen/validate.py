"""Per-clip validation. Every check must pass; failures carry a reason string."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backends.base import ClipResult
from .spec import ScreenMove

MAX_PENETRATION_M = 0.003
MAX_FRAME_JUMP_M = 0.6          # faster than any claw move between frames => teleport
REST_LIN_SPEED = 0.02           # m/s, final frame
REST_ANG_SPEED = 1.0            # rad/s, final frame
MIN_VISIBLE_PX_128 = 6          # every non-reserve object visible at first and last frame (px at 128x128)


def min_visible_px(width: int, height: int) -> int:
    """MIN_VISIBLE_PX_128 scaled by frame area, so the check means the same at any resolution.
    A twin's hidden change must also differ by at least this many seg pixels at the end."""
    return max(1, round(MIN_VISIBLE_PX_128 * width * height / (128 * 128)))


@dataclass
class Validation:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def _px_per_object(seg: np.ndarray, n_obj: int) -> np.ndarray:
    """[T, n_obj] visible pixel count per object."""
    T = seg.shape[0]
    out = np.zeros((T, n_obj), dtype=np.int64)
    for t in range(T):
        c = np.bincount(seg[t].ravel(), minlength=256)
        out[t] = c[1:n_obj + 1]
    return out


def validate_clip(clip: ClipResult, diag: dict) -> Validation:
    ep, st = clip.episode, clip.states
    reasons: list[str] = []
    meta = ep.meta
    first, last = st[0].count_per_region, st[-1].count_per_region
    n_obj = len(ep.objects)
    reserve_ids = {o.id for o in ep.objects if o.start_region == "reserve"}
    v = ep.violation

    # counts
    if first["box"] != meta["n0"]:
        reasons.append(f"initial box count {first['box']} != n0 {meta['n0']}")
    want_final = meta["shown_final"] if v is not None else meta["final"]
    if last["box"] != want_final:
        reasons.append(f"final box count {last['box']} != expected {want_final}")
    if first["hopper"] != meta["n_hopper"]:
        reasons.append(f"initial hopper count {first['hopper']} != {meta['n_hopper']}")
    moved = meta["k"] if meta["op"] == "add" else -meta["k"]
    if last["hopper"] != meta["n_hopper"] - moved:
        reasons.append(f"final hopper count {last['hopper']} != {meta['n_hopper'] - moved}")
    for s in st:
        if s.count_per_region["table"]:
            reasons.append(f"object escaped to table at t={s.t:.2f}")
            break
    exp_reserve_end = len(reserve_ids) + {None: 0, "add": -1, "delete": 1, "move": 0, "swap": 0}[v.kind if v else None]
    if last["reserve"] != exp_reserve_end:
        reasons.append(f"reserve count {last['reserve']} != {exp_reserve_end}")
    if last["claw"]:
        reasons.append("object still held at end")

    # physics quality
    if diag["max_penetration"] > MAX_PENETRATION_M:
        reasons.append(f"penetration {diag['max_penetration'] * 1000:.1f} mm")
    for o in st[-1].objects:
        if o["region"] == "reserve":
            continue
        if np.linalg.norm(o["lin_vel"]) > REST_LIN_SPEED or np.linalg.norm(o["ang_vel"]) > REST_ANG_SPEED:
            reasons.append(f"object {o['id']} not at rest at end")
            break

    # teleport check (the violation's own jump excepted)
    viol_objs = {e["object_id"] for e in clip.events if e["event"].startswith("violation")}
    for a, b in zip(st, st[1:]):
        for oa, ob in zip(a.objects, b.objects):
            if oa["id"] in viol_objs:
                continue
            if np.linalg.norm(np.subtract(ob["pos"], oa["pos"])) > MAX_FRAME_JUMP_M:
                reasons.append(f"object {oa['id']} jumped between t={a.t:.2f} and {b.t:.2f}")

    # visibility
    px = _px_per_object(clip.seg, n_obj)
    MIN_VISIBLE_PX = min_visible_px(ep.clip.width, ep.clip.height)
    for idx, name in ((0, "first"), (-1, "last")):
        for o in st[idx].objects:
            if o["region"] in ("box", "hopper") and px[idx, o["id"]] < MIN_VISIBLE_PX:
                reasons.append(f"object {o['id']} not visible ({px[idx, o['id']]} px) in {name} frame")
    for oid in reserve_ids:
        for ti, s in enumerate(st):
            if s.objects[oid]["region"] == "reserve" and px[ti, oid] > 0:
                reasons.append(f"reserve object {oid} visible at t={s.t:.2f}")
                break
    # every carried object must show while held: the claw's work is the visible cause
    held_px: dict[int, int] = {}
    for ti, s in enumerate(st):
        for o in s.objects:
            if o["region"] == "claw":
                held_px[o["id"]] = max(held_px.get(o["id"], 0), int(px[ti, o["id"]]))
    grasped = [e["object_id"] for e in clip.events if e["event"] == "grasp"]
    for oid in grasped:
        if held_px.get(oid, 0) < MIN_VISIBLE_PX:
            reasons.append(f"carried object {oid} never visible while held ({held_px.get(oid, 0)} px)")
    # the screen must not hide the hopper: check the first frame with the screen fully raised
    first_up = next((ti for ti, s in enumerate(st) if ep.box_occluded_at(s.t)), None)
    if first_up is not None:
        for o in st[first_up].objects:
            if o["region"] == "hopper" and px[first_up, o["id"]] < MIN_VISIBLE_PX and px[0, o["id"]] >= MIN_VISIBLE_PX:
                reasons.append(f"hopper object {o['id']} hidden by the raised screen")
    # occlusion: while the screen is fully raised, nothing resting in the box may show
    for ti, s in enumerate(st):
        if ep.box_occluded_at(s.t):
            leak = [o["id"] for o in s.objects if o["region"] == "box" and px[ti, o["id"]] > 0]
            if leak:
                reasons.append(f"box contents visible behind raised screen at t={s.t:.2f}: {leak}")
                break

    lower = next((a for a in ep.actions if isinstance(a, ScreenMove) and a.kind == "screen_lower"), None)
    metrics = {
        "max_penetration_mm": diag["max_penetration"] * 1000,
        "reveal_t": lower.t_start if lower else None,
        "final_object_px": int(px[-1].sum()),
    }
    return Validation(ok=not reasons, reasons=reasons, metrics=metrics)


def validate_twins(possible: ClipResult, impossible: ClipResult) -> Validation:
    """Twins must be pixel-identical before the violation."""
    v = impossible.episode.violation
    if v is None or impossible.episode.twin_of != possible.episode.episode_id:
        return Validation(False, ["not a twin pair"])
    times = possible.episode.clip.frame_times()
    reasons = []
    for i, t in enumerate(times):
        if t >= v.t:
            break
        if not np.array_equal(possible.frames[i], impossible.frames[i]):
            reasons.append(f"twins differ before violation at frame {i} (t={t:.2f})")
            break
    diff = int((possible.seg[-1] != impossible.seg[-1]).sum())
    if diff < min_visible_px(possible.episode.clip.width, possible.episode.clip.height):
        reasons.append(f"hidden {v.kind} barely visible at the end ({diff} px differ)")
    return Validation(ok=not reasons, reasons=reasons, metrics={"final_seg_diff_px": diff})
