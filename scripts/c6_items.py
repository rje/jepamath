"""C6 items: joint use of the initial count and both transfers (docs/2026-11-26-c6-joint-count-design-draft.md).

Two roles, disjoint seed ranges, written to separate directories. Pairing and the freeze are c6_manifest.py.

  source  one base scene (occluded, n0=2, one add + one shuffle, n_hopper=1, no glides) -> SEVEN clips:
          initial box N in {1, 2} (N=1: one random box object moved to its reserve pose before simulation)
          x add slot (HB or BB) x remove slot (BH or BB); N=1 with remove-only is omitted (it ends empty).
          Final = N + a - r. Same camera, physics, objects, screen, times and per-slot drop hints in all seven.
          add_first (a coin per scene) puts the add slot first or second.
  donor   one base scene from the same template filter -> THREE natural clips, both transfers box->box, starting
          with 1, 2 and 3 objects in the box. Box objects are area-matched across the three: radius
          DONOR_R1 * sqrt(1/c) (+-DONOR_JITTER), so total projected area is about constant. The measured
          post-reveal box-object pixel area per count is saved for the area gate (frozen on the pilot).

Checks: validate_clip on every clip; prefix identity before transfer 1 within each initial state (item_checks per
group); one ctx / first post-reveal slot per source scene; each carried object visible in context AFTER the hopper
mask (the H-C2 arm) at model resolution. Scenes failing any check are logged to rejects.jsonl.

    python scripts/c6_items.py --role source --out DIR --seeds 930000 940000 --n 400
    python scripts/c6_items.py --role donor  --out DIR --seeds 950000 960000 --n 600
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from jepagen.limiter import RenderLimiter  # noqa: E402
from jepagen.run import run_episodes_batched  # noqa: E402
from jepagen.sampler import (RESERVE_Z, Manifest, Template, _place_in, _project, _size_for, _volume,  # noqa: E402
                             _z_extent, load_yaml, sample_episode)
from jepagen.spec import Episode, ObjectSpec, ScreenMove, Transfer  # noqa: E402
from jepagen.validate import min_visible_px, validate_clip  # noqa: E402
from jepatrain.data import downsample  # noqa: E402
from reversal_items import BB, BH, HB, RES, TUBELET, _drop, box_tokens, item_checks  # noqa: E402

# (name, N, add active, remove active)
CELLS = [("N1-", 1, 0, 0), ("N1A", 1, 1, 0), ("N1AR", 1, 1, 1),
         ("N2-", 2, 0, 0), ("N2A", 2, 1, 0), ("N2R", 2, 0, 1), ("N2AR", 2, 1, 1)]
DONOR_COUNTS = (1, 2, 3)
DONOR_R1, DONOR_JITTER = 0.062, 0.02      # r_c = R1/sqrt(c): 0.062, 0.044, 0.036, inside v4b's [0.035, 0.065]
# per-count radius calibration (renderer-only pilot, 64 donors, seeds 980000-999000): measured post-reveal area
# was 1.033 / 0.989 / 0.978 of the donor mean for 1 / 2 / 3 objects (overlap and occlusion grow with count)
DONOR_CAL = {1: 0.984, 2: 1.0055, 3: 1.011}


def base_ok(ep: Episode) -> bool:
    m = ep.meta
    return (m["visibility"] == "occluded" and m["op"] == "add" and m["k"] == 1 and m["shuffles"] == 1
            and m["n0"] == 2 and m["n_hopper"] == 1 and not ep.glides and ep.violation is None
            and sum(isinstance(a, Transfer) for a in ep.actions) == 2)


def retime(base: Episode) -> Episode:
    """Move the reveal later so BOTH transfers, releases included, end inside the model's context: the first
    context-boundary frame (a tubelet multiple) is at least 1.5 frames after the second transfer's end, and the
    screen starts lowering just after it (same lowering duration). Astra review, 26 Nov: with the stock timeline
    the second release fell after the last context frame in 42/49 pilot histories. Applied to every base, sources
    and donors alike, so the timelines stay comparable for pairing."""
    fps = base.clip.fps
    t2 = max(a.t_end for a in base.actions if isinstance(a, Transfer))
    ctx = math.ceil((t2 * fps + 1.5) / TUBELET) * TUBELET
    acts = []
    for a in base.actions:
        if isinstance(a, ScreenMove) and a.kind == "screen_lower":
            dur = a.t_end - a.t_start
            start = max(a.t_start, ctx / fps + 0.01)
            a = a.model_copy(update={"t_start": start, "t_end": start + dur})
        acts.append(a.model_dump())
    return Episode.model_validate({**base.model_dump(), "actions": acts})


def rescale_sources(base: Episode, rng, radius_range=(0.035, 0.065), jitter=0.1) -> Episode:
    """An independent per-scene object scale (the stock area mode sizes objects for the base's final count, which
    is stale once routes change): base radius ~ U(radius_range), per-object jitter, box and hopper objects
    re-placed for their new sizes."""
    b = float(rng.uniform(*radius_range))
    objs, new = base.objects, []
    radii = {o.id: float(np.clip(b * rng.uniform(1 - jitter, 1 + jitter), *radius_range)) for o in objs}
    sizes = {o.id: _size_for(o.shape, radii[o.id]) for o in objs}
    pos = {}
    for region, tray in (("box", base.scene.box), ("hopper", base.scene.hopper)):
        ids = [o.id for o in objs if o.start_region == region]
        for i, pp in zip(ids, _place_in(rng, tray, [(objs[i].shape, sizes[i], radii[i]) for i in ids])):
            pos[i] = pp
    for o in objs:
        o2 = o.model_copy(update={"size": sizes[o.id], "mass": float(_volume(o.shape, sizes[o.id]) * 600)})
        o2 = o2.model_copy(update={"start_pos": pos[o.id]}) if o.id in pos else _to_reserve(o2)
        new.append(o2)
    return Episode.model_validate({**base.model_dump(), "objects": [o.model_dump() for o in new],
                                   "meta": dict(base.meta, base_radius=b, control_mode="c6_source_scale")})


def _to_reserve(o: ObjectSpec) -> ObjectSpec:
    return o.model_copy(update={"start_region": "reserve",
                                "start_pos": (o.id * 0.3, 0.0, RESERVE_Z + _z_extent(o.shape, o.size) + 0.002)})


def _with(base: Episode, objects, routes, hints, meta, eid) -> Episode:
    acts, i = [], 0
    for a in base.actions:
        if isinstance(a, Transfer):
            src, dst = routes[i]
            acts.append(a.model_copy(update={"src": src, "dst": dst, "drop_xy": hints[i][dst]}).model_dump())
            i += 1
        else:
            acts.append(a.model_dump())
    return Episode.model_validate({**base.model_dump(), "episode_id": eid, "objects": [o.model_dump() for o in objects],
                                   "actions": acts, "meta": meta})


def _net(routes) -> tuple[int, int, str, int]:
    n_in, n_out = sum(r == HB for r in routes), sum(r == BH for r in routes)
    d = n_in - n_out
    return n_in, n_out, ("add" if d > 0 else "remove" if d < 0 else "none"), abs(d)


def make_sources(base: Episode, seed: int) -> tuple[dict[str, Episode], bool]:
    rng = np.random.default_rng([seed, 66])
    base = rescale_sources(retime(base), rng)
    hints = [{"box": _drop(rng, base.scene.box), "hopper": _drop(rng, base.scene.hopper)} for _ in range(2)]
    add_first = bool(seed % 2)            # deterministic, so the two slot orders come out near half and half
    box_ids = [o.id for o in base.objects if o.start_region == "box"]
    drop_id = int(rng.choice(box_ids))
    out = {}
    for name, N, a, r in CELLS:
        objs = [(_to_reserve(o) if (N == 1 and o.id == drop_id) else o) for o in base.objects]
        add_slot, rem_slot = (HB if a else BB), (BH if r else BB)
        routes = (add_slot, rem_slot) if add_first else (rem_slot, add_slot)
        n_in, n_out, op, k = _net(routes)
        meta = dict(base.meta, n0=N, op=op, k=k, final=N + a - r, shuffles=sum(x == BB for x in routes), n_in=n_in,
                    n_out=n_out, cell=name, add_active=a, remove_active=r, add_first=add_first,
                    route=[list(x) for x in routes], scene_seed=seed, dropped_object=drop_id if N == 1 else None)
        out[name] = _with(base, objs, routes, hints, meta, f"c6s-{seed:09d}-{name}")
    return out, add_first


def make_donors(base: Episode, seed: int) -> dict[int, Episode]:
    """Three box objects (shapes/colours from base objects that are not the hopper object), area-matched per count."""
    rng = np.random.default_rng([seed, 77])
    base = retime(base)
    hop = [o for o in base.objects if o.start_region == "hopper"]
    others = [o for o in base.objects if o.start_region != "hopper"][:3]
    if len(others) < 3 or len(hop) != 1:
        raise ValueError(f"{base.episode_id}: need 3 non-hopper objects and 1 hopper object")
    hints = [{"box": _drop(rng, base.scene.box), "hopper": _drop(rng, base.scene.hopper)} for _ in range(2)]
    out = {}
    for c in DONOR_COUNTS:
        rc = DONOR_R1 / math.sqrt(c) * DONOR_CAL[c]
        radii = [float(rc * rng.uniform(1 - DONOR_JITTER, 1 + DONOR_JITTER)) for _ in range(3)]
        sizes = [_size_for(o.shape, r) for o, r in zip(others, radii)]
        pos = _place_in(rng, base.scene.box, [(o.shape, s, r) for o, s, r in zip(others[:c], sizes[:c], radii[:c])])
        objs = []
        for j, (o, s) in enumerate(zip(others, sizes)):
            o2 = o.model_copy(update={"id": j, "size": s, "mass": float(_volume(o.shape, s) * 600)})
            objs.append(o2.model_copy(update={"start_region": "box", "start_pos": pos[j]}) if j < c else _to_reserve(o2))
        objs.append(hop[0].model_copy(update={"id": 3}))
        routes = (BB, BB)
        meta = dict(base.meta, n0=c, op="none", k=0, final=c, shuffles=2, n_in=0, n_out=0, n_hopper=1,
                    route=[list(x) for x in routes], scene_seed=seed, donor_radii=radii[:c], base_radius=rc,
                    control_mode="area_matched_c6")
        out[c] = _with(base, objs, routes, hints, meta, f"c6d-{seed:09d}-{c}")
    return out


def hopper_px(ep: Episode, res: int, patch: int) -> tuple[np.ndarray, np.ndarray]:
    """reversal_score.hopper_masks from an Episode at any resolution (dilation = one model patch)."""
    from matplotlib.path import Path
    from scipy.ndimage import binary_dilation
    from scipy.spatial import ConvexHull
    h, cam, c = ep.scene.hopper, ep.scene.camera, ep.clip
    hx, hy = h.inner_size[0] / 2 + h.wall_thickness, h.inner_size[1] / 2 + h.wall_thickness
    pts = np.array([(h.center[0] + sx * hx, h.center[1] + sy * hy, z) for sx in (-1, 1) for sy in (-1, 1)
                    for z in (0.0, h.wall_height + 0.14)])
    ndc = _project(cam, c.width / c.height, pts)
    px = np.stack([(ndc[:, 0] + 1) / 2 * res, (1 - ndc[:, 1]) / 2 * res], 1)
    yy, xx = np.mgrid[0:res, 0:res]
    inside = Path(px[ConvexHull(px).vertices]).contains_points(np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], 1))
    m = binary_dilation(inside.reshape(res, res), iterations=patch)
    return m, binary_dilation(m, iterations=patch) & ~m


def _tray_px(ep: Episode, tray, res: int) -> np.ndarray:
    c = ep.clip
    pt = np.array([(tray.center[0], tray.center[1], tray.wall_height)])
    ndc = _project(ep.scene.camera, c.width / c.height, pt)[0]
    return np.array([(ndc[0] + 1) / 2 * res, (1 - ndc[1]) / 2 * res])


def transfer_records(clip, ctx: int) -> tuple[list[dict], list[str]]:
    """Per transfer, from the simulation itself: grasped object, its region just before the grasp and once settled
    after the release (must equal the scheduled src / dst), and at MODEL resolution (128 px, hopper mask applied)
    the carried object's pixels outside the mask over the context frames of the carry, plus the image positions
    where it is first and last seen (must be nearer the src / dst tray than the other tray)."""
    ep, st = clip.episode, clip.states
    times = np.array([s.t for s in st])
    s_ = clip.seg.shape[1] // RES
    seg = clip.seg[:, ::s_, ::s_]
    m, _ = hopper_px(ep, RES, 8)
    mv = min_visible_px(RES, RES)
    trays = {"box": _tray_px(ep, ep.scene.box, RES), "hopper": _tray_px(ep, ep.scene.hopper, RES)}
    tr = [a for a in ep.actions if isinstance(a, Transfer)]
    grasps = [(e["t"], e["object_id"]) for e in clip.events if e["event"] == "grasp"]
    rels = [e["t"] for e in clip.events if e["event"] == "release"]
    reasons, recs = [], []
    if len(grasps) != len(tr) or len(rels) != len(tr):
        return [], [f"{len(grasps)} grasps / {len(rels)} releases for {len(tr)} transfers"]
    for k, (a, (tg, oid), trl) in enumerate(zip(tr, grasps, rels)):
        before = [s for s in st if s.t < tg][-1]
        after = [s for s in st if s.t >= trl + 0.3]
        reg0 = next(o["region"] for o in before.objects if o["id"] == oid)
        reg1 = next(o["region"] for o in after[0].objects if o["id"] == oid) if after else None
        if (reg0, reg1) != (a.src, a.dst):
            reasons.append(f"transfer {k}: object {oid} went {reg0}->{reg1}, scheduled {a.src}->{a.dst}")
        if trl >= ctx / ep.clip.fps - 1 / ep.clip.fps:
            reasons.append(f"transfer {k}: release at {trl:.2f}s is not inside the context ({ctx} frames)")
        frames = [i for i in range(min(ctx, len(seg))) if tg <= times[i] <= trl]
        vis = [(i, (seg[i] == oid + 1) & ~m) for i in frames]
        vis = [(i, v) for i, v in vis if v.sum() >= mv]
        rec = {"object": int(oid), "src": a.src, "dst": a.dst, "region_before": reg0, "region_after": reg1,
               "t_grasp": float(tg), "t_release": float(trl), "visible_frames": [i for i, _ in vis]}
        if not vis:
            reasons.append(f"transfer {k}: object {oid} never visible outside the hopper mask in context at 128 px")
        else:
            def cen(v):
                yy, xx = np.nonzero(v)
                return np.array([xx.mean(), yy.mean()])
            c0, c1 = cen(vis[0][1]), cen(vis[-1][1])
            d = lambda c, r: float(np.linalg.norm(c - trays[r]))
            other = lambda r: "hopper" if r == "box" else "box"
            rec.update(first_seen=c0.tolist(), last_seen=c1.tolist())
            if a.src != a.dst:
                if not d(c0, a.src) < d(c0, other(a.src)):
                    reasons.append(f"transfer {k}: first seen nearer {other(a.src)} than its source {a.src}")
                if not d(c1, a.dst) < d(c1, other(a.dst)):
                    reasons.append(f"transfer {k}: last seen nearer {other(a.dst)} than its destination {a.dst}")
        recs.append(rec)
    return recs, reasons


def patch_coverage(clip, first_model: int) -> np.ndarray:
    """[16, 16] mean fraction of each model patch covered by box-start objects over the post-reveal frames
    (128 px seg)."""
    ids = [o.id + 1 for o in clip.episode.objects if o.start_region == "box"]
    s_ = clip.seg.shape[1] // RES
    seg = clip.seg[first_model:, ::s_, ::s_]
    cov = np.isin(seg, ids).astype(float).mean(0)
    return cov.reshape(16, 8, 16, 8).mean((1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=("source", "donor"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=os.path.join(HERE, "..", "configs", "v4b_small.yaml"))
    ap.add_argument("--seeds", type=int, nargs=2, required=True)
    ap.add_argument("--n", type=int, required=True, help="accepted scenes wanted")
    ap.add_argument("--max-render-fps", type=float, default=60.0)
    ap.add_argument("--batch", type=int, default=8, help="scenes per GPU submission")
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit(f"{a.out} is not empty: refusing to append")
    os.makedirs(a.out, exist_ok=True)
    t, man = load_yaml(a.template, Template), Manifest()
    lim = RenderLimiter(os.path.join(a.out, ".limiter"), a.max_render_fps)
    log = open(os.path.join(a.out, "items.jsonl"), "a")
    rej = open(os.path.join(a.out, "rejects.jsonl"), "a")
    kept, pending = 0, []

    def flush():
        nonlocal kept
        eps = [e for _, h, _ in pending for e in h.values()]
        res = run_episodes_batched(eps, lim)
        j = 0
        for seed, h, extra in pending:
            clips, reasons = {}, []
            for k in h:
                clip, diag = res[j]; j += 1
                v = validate_clip(clip, diag)
                if not v.ok:
                    reasons += [f"{k}: {r}" for r in v.reasons]
                clips[k] = clip
            fps = next(iter(h.values())).clip.fps
            info = None
            if a.role == "source":
                groups = [{k: clips[k] for k in clips if h[k].meta["n0"] == N} for N in (1, 2)]
            else:
                groups = [{k: clips[k]} for k in clips]
            for g in groups:
                r2, inf = item_checks(g, fps)
                if a.role == "donor":       # donors only need the timing; carry visibility is the source's concern
                    r2 = [r for r in r2 if "not visible in context" not in r]
                reasons += r2
                if info is not None and inf != info:
                    reasons.append(f"timing differs between groups: {inf} vs {info}")
                info = info or inf
            n_frames = next(iter(clips.values())).frames.shape[0]
            trecs = {}
            if not reasons:
                for k in clips:
                    tr_, r3 = transfer_records(clips[k], info["ctx"])
                    trecs[str(k)] = tr_
                    if a.role == "source":
                        reasons += [f"{k}: {r}" for r in r3]
                    else:   # donors: only the transfer-region and in-context checks matter (BB, BB)
                        reasons += [f"{k}: {r}" for r in r3 if "went" in r or "not inside" in r]
            if reasons or kept >= a.n:
                if reasons:
                    rej.write(json.dumps({"seed": seed, "reasons": reasons}) + "\n"); rej.flush()
                continue
            names = list(h)
            ep0 = h[names[0]]
            m, ring = hopper_px(ep0, RES, 8)
            rec = {"seed": seed, "role": a.role, "names": [str(x) for x in names], **info,
                   "n_frames": int(n_frames), "meta": {str(k): h[k].meta for k in names},
                   "events": {str(k): clips[k].events for k in names}, "transfers": trecs}
            arrays = dict(frames=np.stack([downsample(clips[k].frames, RES) for k in names]),
                          box_tokens=box_tokens(ep0), hopper_px=m, hopper_ring=ring)
            if a.role == "source":
                rec["add_first"] = extra
            else:
                first_model = info["ctx"] + info["first_post_slot"] * TUBELET
                arrays["coverage"] = np.stack([patch_coverage(clips[k], first_model) for k in names])
                ar = arrays["coverage"].reshape(len(names), -1).sum(1)
                rec["ending_area_px"] = {str(k): float(v) * 64 for k, v in zip(names, ar)}
                rec["area_ratio"] = float(ar.max() / max(ar.min(), 1e-9))
            np.savez_compressed(os.path.join(a.out, f"{a.role}_{seed:09d}.npz"), **arrays)
            rec["episodes"] = {str(k): h[k].model_dump() for k in names}
            log.write(json.dumps(rec, default=float) + "\n"); log.flush()
            kept += 1
        pending.clear()

    for seed in range(*a.seeds):
        if kept >= a.n:
            break
        base = sample_episode(t, man, "train", seed)
        if not base_ok(base):
            continue
        try:
            if a.role == "source":
                h, extra = make_sources(base, seed)
            else:
                h, extra = make_donors(base, seed), None
        except ValueError as ex:
            rej.write(json.dumps({"seed": seed, "reasons": [f"build: {ex}"]}) + "\n"); rej.flush()
            continue
        pending.append((seed, h, extra))
        if len(pending) == a.batch:
            flush()
            print(f"seed {seed}: kept {kept}", flush=True)
    if pending:
        flush()
    print(json.dumps({"kept": kept}))
    if kept < a.n:
        raise SystemExit(f"seed range exhausted before {a.n} scenes: {kept}")


if __name__ == "__main__":
    main()
