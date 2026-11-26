"""C9 items (DRAFT design: docs/2026-11-27-c9-number-vs-area-draft.md): source-side number vs carried area / volume.

source  one base scene (C6 base filter, retimed) -> 12 clips: 3 families x 4 histories, one shape per block
        (sphere / box / cylinder by seed mod 3). Objects: id0 = a (box, left), id1 = second box object (right),
        id2 = the hopper object, id3 = reserve. Sizes a=.035, s=.037, b=.062.
          H        box {a, s}  hopper {s_in}   BB, HB   -> {a, s, s_in}  final 3   (grasps 1, 2)
          L-N      box {a}     hopper {b_low}  BB, HB   -> {a, b_low}    final 2   (grasps 0, 2)
          L-add    box {a, b_low} hopper {s}   BB, BB   -> {a, b_low}    final 2   (grasps 1, 1)
          L-remove box {a, s}  hopper {b_low}  BH, HB   -> {a, b_low}    final 2   (grasps 1, 2)
        families (b_low, s_in): orig (b, s); area (sqrt2 s, sqrt(b^2-s^2)); volume (cbrt2 s, cbrt(b^3-s^3)).
        In 'area' the final summed area of H and the L histories is exactly the orig values swapped; same for
        volume in 'volume' (all sizes scale every dimension with r, for every shape).
donor   one base scene -> 2 clips (2 and 3 objects in the box, both transfers box->box) in one of two layouts:
          'rev'  2-ending: 2 objects r=.061 placed wide;   3-ending: 3 objects r=.037 compact
          'fwd'  2-ending: 2 objects r=.037 compact;       3-ending: 3 objects r=.061 wide
        Saved: post-reveal coverage per patch, measured spread; metadata area / volume / radii from the spec.

Checks: validate_clip; simulator transfer records (actual src / dst, release in context, masked visibility and
src/dst cues at 128 px, from c6_items.transfer_records); expected grasp ids per history; one ctx / first post slot
per block; sources: hopper-masked frames before transfer 1 identical between L-remove and H in each family.

    python scripts/c9_items.py --role source --out DIR --seeds A B --n N
    python scripts/c9_items.py --role donor --layout rev --out DIR --seeds A B --n N
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
from c6_items import (BB, BH, HB, RES, TUBELET, _to_reserve, _with, base_ok, hopper_px, item_checks,  # noqa: E402
                      patch_coverage, retime, transfer_records)
from jepagen.limiter import RenderLimiter  # noqa: E402
from jepagen.run import run_episodes_batched  # noqa: E402
from jepagen.sampler import Manifest, Template, _size_for, _volume, _z_extent, load_yaml, sample_episode  # noqa: E402
from jepagen.spec import Episode, Transfer  # noqa: E402
from jepagen.validate import validate_clip  # noqa: E402
from jepatrain.data import downsample  # noqa: E402
from reversal_items import _drop, box_tokens  # noqa: E402

A, S_, B_ = 0.035, 0.037, 0.062
FAMILIES = {"orig": (B_, S_),
            "area": (math.sqrt(2) * S_, math.sqrt(B_ ** 2 - S_ ** 2)),
            "volume": (2 ** (1 / 3) * S_, (B_ ** 3 - S_ ** 3) ** (1 / 3))}
HISTORIES = ("H", "L-N", "L-add", "L-remove")
GRASPS = {"H": [1, 2], "L-N": [0, 2], "L-add": [1, 1], "L-remove": [1, 2]}
SHAPES = ("sphere", "box", "cylinder")
DONOR_BIG, DONOR_SMALL = 0.061, 0.037


def _obj(o, shape, r, region, pos):
    size = _size_for(shape, r)
    o2 = o.model_copy(update={"shape": shape, "size": size, "mass": float(_volume(shape, size) * 600)})
    if region == "reserve":
        return _to_reserve(o2)
    return o2.model_copy(update={"start_region": region, "start_pos": (pos[0], pos[1], pos[2] + _z_extent(shape, size))})


def _floor(tray, x, y):
    return (float(x), float(y), tray.wall_thickness + 0.004)


def make_sources(base: Episode, seed: int) -> dict[str, Episode]:
    base = retime(base)
    rng = np.random.default_rng([seed, 99])
    shape = SHAPES[seed % 3]
    bx, hp = base.scene.box, base.scene.hopper
    dx = bx.inner_size[0] / 4
    left, right = _floor(bx, bx.center[0] - dx, bx.center[1]), _floor(bx, bx.center[0] + dx, bx.center[1])
    hop = _floor(hp, hp.center[0], hp.center[1])
    hints = [{"box": _drop(rng, bx), "hopper": _drop(rng, hp)} for _ in range(2)]
    o = sorted(base.objects, key=lambda x: x.id)
    if len(o) < 4:
        raise ValueError("need 4 base objects")
    out = {}
    for fam, (b_low, s_in) in FAMILIES.items():
        for h in HISTORIES:
            if h == "H":
                objs = [_obj(o[0], shape, A, "box", left), _obj(o[1], shape, S_, "box", right),
                        _obj(o[2], shape, s_in, "hopper", hop)]
                routes, n0 = (BB, HB), 2
            elif h == "L-N":
                objs = [_obj(o[0], shape, A, "box", left), _obj(o[1], shape, S_, "reserve", None),
                        _obj(o[2], shape, b_low, "hopper", hop)]
                routes, n0 = (BB, HB), 1
            elif h == "L-add":
                objs = [_obj(o[0], shape, A, "box", left), _obj(o[1], shape, b_low, "box", right),
                        _obj(o[2], shape, S_, "hopper", hop)]
                routes, n0 = (BB, BB), 2
            else:
                objs = [_obj(o[0], shape, A, "box", left), _obj(o[1], shape, S_, "box", right),
                        _obj(o[2], shape, b_low, "hopper", hop)]
                routes, n0 = (BH, HB), 2
            objs.append(_obj(o[3], shape, S_, "reserve", None))
            n_in, n_out = sum(r == HB for r in routes), sum(r == BH for r in routes)
            d = n_in - n_out
            final = n0 + d
            meta = dict(base.meta, n0=n0, n_hopper=1, op="add" if d > 0 else "remove" if d < 0 else "none", k=abs(d),
                        final=final, shuffles=sum(r == BB for r in routes), route=[list(r) for r in routes],
                        history=h, family=fam, shape=shape, scene_seed=seed,
                        radii={"a": A, "s": S_, "b_low": b_low, "s_in": s_in})
            out[f"{fam}:{h}"] = _with(base, objs, routes, hints, meta, f"c9s-{seed:09d}-{fam}-{h}")
    return out


def make_donors(base: Episode, seed: int, layout: str) -> dict[int, Episode]:
    base = retime(base)
    rng = np.random.default_rng([seed, 88])
    shape = SHAPES[seed % 3]
    bx, hp = base.scene.box, base.scene.hopper
    o = sorted(base.objects, key=lambda x: x.id)
    hop_obj = next(x for x in o if x.start_region == "hopper")
    others = [x for x in o if x.start_region != "hopper"][:3]
    hx, hy = bx.inner_size[0] / 2, bx.inner_size[1] / 2
    out = {}
    for c in (2, 3):
        wide = (layout == "rev") == (c == 2)        # rev: the 2-ending is big+wide; fwd: the 3-ending is
        r = DONOR_BIG if wide else DONOR_SMALL
        if wide:
            m = r + 0.012
            spots = [(bx.center[0] - hx + m, bx.center[1] - hy + m), (bx.center[0] + hx - m, bx.center[1] + hy - m),
                     (bx.center[0] - hx + m, bx.center[1] + hy - m)][:c]
            hint_xy = [spots[0], spots[1]]
        else:
            g = 2 * r + 0.006
            spots = [(bx.center[0] - g / 2, bx.center[1]), (bx.center[0] + g / 2, bx.center[1]),
                     (bx.center[0], bx.center[1] + g * 0.87)][:c]
            hint_xy = [(bx.center[0], bx.center[1]), (bx.center[0], bx.center[1])]
        objs = []
        for j, oo in enumerate(others):
            objs.append(_obj(oo.model_copy(update={"id": j}), shape, r, "box" if j < c else "reserve",
                             _floor(bx, *spots[j]) if j < c else None))
        objs.append(hop_obj.model_copy(update={"id": 3}))
        hints = [{"box": tuple(map(float, hint_xy[i])), "hopper": _drop(rng, hp)} for i in range(2)]
        meta = dict(base.meta, n0=c, op="none", k=0, final=c, shuffles=2, n_hopper=1, route=[list(BB), list(BB)],
                    layout=layout, wide=wide, shape=shape, scene_seed=seed, donor_radius=r)
        out[c] = _with(base, objs, (BB, BB), hints, meta, f"c9d-{seed:09d}-{layout}-{c}")
    return out


def spread(cov: np.ndarray) -> float:
    """Coverage-weighted RMS distance (patches) from the coverage centroid, over the [16, 16] map."""
    yy, xx = np.mgrid[0:16, 0:16]
    w = cov / max(cov.sum(), 1e-12)
    cx, cy = (w * xx).sum(), (w * yy).sum()
    return float(np.sqrt((w * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum()))


def masked_prefix_equal(c1, c2, t1: float) -> bool:
    ep = c1.episode
    m, _ = hopper_px(ep, RES, 8)
    times = ep.clip.frame_times()
    f1, f2 = downsample(c1.frames, RES), downsample(c2.frames, RES)
    for i, t in enumerate(times):
        if t >= t1:
            break
        if not np.array_equal(f1[i][~m], f2[i][~m]):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=("source", "donor"), required=True)
    ap.add_argument("--layout", choices=("rev", "fwd"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=os.path.join(HERE, "..", "configs", "v4b_small.yaml"))
    ap.add_argument("--seeds", type=int, nargs=2, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--max-render-fps", type=float, default=60.0)
    ap.add_argument("--batch", type=int, default=6)
    a = ap.parse_args()
    if a.role == "donor" and not a.layout:
        raise SystemExit("--layout is required for donors")
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
        res = run_episodes_batched([e for _, h in pending for e in h.values()], lim)
        j = 0
        for seed, h in pending:
            clips, reasons = {}, []
            for k in h:
                clip, diag = res[j]; j += 1
                v = validate_clip(clip, diag)
                reasons += [f"{k}: {r}" for r in v.reasons]
                clips[k] = clip
            fps = next(iter(h.values())).clip.fps
            infos = set()
            for k, c in clips.items():
                r2, inf = item_checks({k: c}, fps)
                reasons += [f"{k}: {r}" for r in r2 if "not visible in context" not in r and "grasps" not in r]
                infos.add((inf["ctx"], inf["first_post_slot"]))
            if len(infos) != 1:
                reasons.append(f"timing differs across clips: {infos}")
            ctx, first = next(iter(infos))
            trecs = {}
            if not reasons:
                for k, c in clips.items():
                    tr, r3 = transfer_records(c, ctx)
                    trecs[k] = tr
                    if a.role == "source":
                        reasons += [f"{k}: {r}" for r in r3]
                        got = [x["object"] for x in tr]
                        want = GRASPS[h[k].meta["history"]]
                        if got != want:
                            reasons.append(f"{k}: grasped {got}, expected {want}")
                    else:
                        reasons += [f"{k}: {r}" for r in r3 if "went" in r or "not inside" in r]
            if a.role == "source" and not reasons:
                t1 = min(x.t_start for x in h["orig:H"].actions if isinstance(x, Transfer))
                for fam in FAMILIES:
                    if not masked_prefix_equal(clips[f"{fam}:L-remove"], clips[f"{fam}:H"], t1):
                        reasons.append(f"{fam}: L-remove and H differ outside the hopper mask before transfer 1")
            if reasons or kept >= a.n:
                if reasons:
                    rej.write(json.dumps({"seed": seed, "reasons": reasons}) + "\n"); rej.flush()
                continue
            names = list(h)
            ep0 = h[names[0]]
            m, ring = hopper_px(ep0, RES, 8)
            rec = {"seed": seed, "role": a.role, "names": names, "ctx": ctx, "first_post_slot": first,
                   "n_frames": int(next(iter(clips.values())).frames.shape[0]),
                   "meta": {k: h[k].meta for k in names}, "transfers": trecs}
            arrays = dict(frames=np.stack([downsample(clips[k].frames, RES) for k in names]),
                          box_tokens=box_tokens(ep0), hopper_px=m, hopper_ring=ring)
            if a.role == "donor":
                arrays["coverage"] = np.stack([patch_coverage(clips[k], ctx + first * TUBELET) for k in names])
                rec["spread"] = {str(k): spread(cv) for k, cv in zip(names, arrays["coverage"])}
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
            h = make_sources(base, seed) if a.role == "source" else make_donors(base, seed, a.layout)
        except ValueError as ex:
            rej.write(json.dumps({"seed": seed, "reasons": [f"build: {ex}"]}) + "\n"); rej.flush()
            continue
        pending.append((seed, h))
        if len(pending) == a.batch:
            flush()
            print(f"seed {seed}: kept {kept}", flush=True)
    if pending:
        flush()
    print(json.dumps({"kept": kept}))


if __name__ == "__main__":
    main()
