"""C10 items (docs/2026-11-27-c10-count-vs-route-size.md): one base scene -> 17 histories crossing the starting count
n0 with the two-move route, bimodal random object sizes, so that route, start count and carried size each disagree
with the final count on some within-scene pairs.

  net 0 : BB-BB, HB-BH, BH-HB   x n0 in {1, 2, 3}
  net +1: BB-HB, HB-BB          x n0 in {1, 2}
  net -1: BH-BB, BB-BH          x n0 in {2, 3}
Objects: id0-2 box pool at fixed slots (n0 uses the first n0, the rest in reserve), id3 hopper, id4 reserve. Sizes per object AND per
history (amendment 1): ONE class per history, small U(.033, .038) or big U(.058, .062), p = 1/2. One shape per scene (seed mod 3). The claw takes the
tallest object, so final box contents come from the simulation (c9_score.final_objects), not the schedule.
Checks as C9 sources (validate_clip, item_checks, transfer_records), one ctx / first post slot per scene.

    python scripts/c10_items.py --out DIR --seeds 8000000 8500000 --per-shape 80
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from c6_items import BB, BH, HB, RES, _with, base_ok, hopper_px, item_checks, retime, transfer_records  # noqa: E402
from c9_items import SHAPES, _floor, _obj  # noqa: E402
from jepagen.limiter import RenderLimiter  # noqa: E402
from jepagen.run import run_episodes_batched  # noqa: E402
from jepagen.sampler import Manifest, Template, load_yaml, sample_episode  # noqa: E402
from jepagen.validate import validate_clip  # noqa: E402
from jepatrain.data import downsample  # noqa: E402
from reversal_items import _drop, box_tokens  # noqa: E402

SMALL, BIG = (0.033, 0.038), (0.058, 0.062)   # 3 small < 2 big in radius sum, area and volume
R = {"BB": BB, "HB": HB, "BH": BH}
ROUTES = {"BB-BB": 0, "HB-BH": 0, "BH-HB": 0, "BB-HB": 1, "HB-BB": 1, "BH-BB": -1, "BB-BH": -1}
N0 = {0: (1, 2, 3), 1: (1, 2), -1: (2, 3)}
HISTORIES = [(n0, r) for r, net in ROUTES.items() for n0 in N0[net]]          # 9 + 4 + 4 = 17


def name(n0, r):
    return f"n{n0}:{r}"


def make_sources(base, seed: int) -> dict:
    base = retime(base)
    rng = np.random.default_rng([seed, 1010])
    shape = SHAPES[seed % 3]
    bx, hp = base.scene.box, base.scene.hopper
    slots = [_floor(bx, bx.center[0] - 0.07, bx.center[1] - 0.045), _floor(bx, bx.center[0] + 0.07, bx.center[1] - 0.045),
             _floor(bx, bx.center[0], bx.center[1] + 0.065)]
    hop = _floor(hp, hp.center[0], hp.center[1])
    hints = [{"box": _drop(rng, bx), "hopper": _drop(rng, hp)} for _ in range(2)]
    o = sorted(base.objects, key=lambda x: x.id)
    if len(o) < 4:
        raise ValueError("need 4 base objects")
    protos = [o[i % len(o)].model_copy(update={"id": i}) for i in range(5)]
    out = {}
    for n0, r in HISTORIES:
        # sizes drawn per HISTORY (amendment 1): with one shared pool, 3- and 2-object endings always overlap and a
        # size-opposed pair (E4) cannot occur
        # one size class per history (as v4b: one base radius per clip), independent of n0 and route
        band = BIG if rng.uniform() < 0.5 else SMALL
        radii = [float(rng.uniform(*band)) for _ in range(4)]
        routes = tuple(R[x] for x in r.split("-"))
        objs = [_obj(protos[i], shape, radii[i], "box" if i < n0 else "reserve", slots[i] if i < n0 else None)
                for i in range(3)]
        objs += [_obj(protos[3], shape, radii[3], "hopper", hop), _obj(protos[4], shape, SMALL[0], "reserve", None)]
        net = ROUTES[r]
        meta = dict(base.meta, n0=n0, n_hopper=1, op="add" if net > 0 else "remove" if net < 0 else "none", k=abs(net),
                    final=n0 + net, shuffles=sum(x == BB for x in routes), route=[list(x) for x in routes],
                    route_name=r, net=net, shape=shape, scene_seed=seed, radii=radii)
        out[name(n0, r)] = _with(base, objs, routes, hints, meta, f"c10-{seed:09d}-n{n0}-{r}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=os.path.join(HERE, "..", "configs", "v4b_small.yaml"))
    ap.add_argument("--seeds", type=int, nargs=2, required=True)
    ap.add_argument("--per-shape", type=int, required=True)
    ap.add_argument("--max-render-fps", type=float, default=60.0)
    ap.add_argument("--batch", type=int, default=3, help="scenes per render batch (17 clips each; engine max 64)")
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit(f"{a.out} is not empty: refusing to append")
    os.makedirs(a.out, exist_ok=True)
    t, man = load_yaml(a.template, Template), Manifest()
    lim = RenderLimiter(os.path.join(a.out, ".limiter"), a.max_render_fps)
    log = open(os.path.join(a.out, "items.jsonl"), "a")
    rej = open(os.path.join(a.out, "rejects.jsonl"), "a")
    kept = {s: 0 for s in SHAPES}
    pending = []

    def flush():
        try:
            res = run_episodes_batched([e for _, h in pending for e in h.values()], lim)
            per, j = {}, 0
            for seed, h in pending:
                per[seed] = res[j:j + len(h)]; j += len(h)
        except RuntimeError:
            per = {}
            for seed, h in pending:
                try:
                    per[seed] = run_episodes_batched(list(h.values()), lim)
                except RuntimeError as ex:
                    rej.write(json.dumps({"seed": seed, "reasons": [f"simulator: {ex}"]}) + "\n"); rej.flush()
        for seed, h in pending:
            if seed not in per:
                continue
            shape = SHAPES[seed % 3]
            clips, reasons, infos = {}, [], set()
            for k, (clip, diag) in zip(h, per[seed]):
                clips[k] = clip
                reasons += [f"{k}: {x}" for x in validate_clip(clip, diag).reasons]
                r2, inf = item_checks({k: clip}, clip.episode.clip.fps)
                reasons += [f"{k}: {x}" for x in r2 if "not visible in context" not in x and "grasps" not in x]
                infos.add((inf["ctx"], inf["first_post_slot"]))
            if len(infos) != 1:
                reasons.append(f"timing differs across clips: {infos}")
            trecs = {}
            if not reasons:
                ctx, first = next(iter(infos))
                for k, c in clips.items():
                    tr, r3 = transfer_records(c, ctx)
                    trecs[k] = tr
                    reasons += [f"{k}: {x}" for x in r3]
            if reasons or kept[shape] >= a.per_shape:
                if reasons:
                    rej.write(json.dumps({"seed": seed, "reasons": reasons[:20]}) + "\n"); rej.flush()
                continue
            ctx, first = next(iter(infos))
            names = list(h)
            ep0 = h[names[0]]
            m, ring = hopper_px(ep0, RES, 8)
            rec = {"seed": seed, "role": "source", "names": names, "ctx": ctx, "first_post_slot": first,
                   "n_frames": int(clips[names[0]].frames.shape[0]), "meta": {k: h[k].meta for k in names},
                   "transfers": trecs, "episodes": {k: h[k].model_dump() for k in names}}
            np.savez_compressed(os.path.join(a.out, f"source_{seed:09d}.npz"),
                                frames=np.stack([downsample(clips[k].frames, RES) for k in names]),
                                box_tokens=box_tokens(ep0), hopper_px=m, hopper_ring=ring)
            log.write(json.dumps(rec, default=float) + "\n"); log.flush()
            kept[shape] += 1
        pending.clear()

    for seed in range(*a.seeds):
        if all(v >= a.per_shape for v in kept.values()):
            break
        if kept[SHAPES[seed % 3]] >= a.per_shape:
            continue
        base = sample_episode(t, man, "train", seed)
        if not base_ok(base):
            continue
        try:
            h = make_sources(base, seed)
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
    if any(v < a.per_shape for v in kept.values()):
        raise SystemExit(f"short of {a.per_shape} per shape: {kept}")


if __name__ == "__main__":
    main()
