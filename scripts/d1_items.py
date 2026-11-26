"""D1 training clips (docs/2026-11-27-d1-number-size-finetune.md): matched two-transfer histories in which the
final count and the carried size agree (concordant) or disagree (discordant).

One v4b base scene (2 transfers, any visibility, stock timing) -> a matched pair of histories with one shape per
scene and the base's colours, screen schedule and camera:
  add (n0 in {0, 1}; box {a} or a held in reserve; hopper always holds 2 objects):
    two  HB, HB        -> n0 + 2     hopper {X1, X2} both carried
    one  HB + BB       -> n0 + 1     hopper {Y, f}: Y carried (taller), filler f stays; order random (HB first at n0=0)
      concordant  X = big,   Y = small    (more objects, more area)
      discordant  X = small, Y = big      (more objects, LESS area and volume: 2 s^2 < b^2, 2 s^3 < b^3)
  remove (box {a, s, b} or {s, b}; hopper {f}): always concordant (the claw takes the tallest object first)
    two  BH, BH -> n0 - 2        one  BH + BB (order random) -> n0 - 1
Sizes: a ~ U(.035, .065); small ~ U(.035, .040); big ~ U(.058, .065).
Subsets: A = add scenes rendered BOTH ways (A_conc/, A_disc/), B = add scenes concordant only, R = remove scenes.
Checks: validate_clip; grasp/release counts; final box contents by size class from the last simulated state;
the discordant area/volume inequality from the spec.

    python scripts/d1_items.py --subset A --out DIR --seeds 6000000 6100000 --n 1000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import webdataset as wds

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from c6_items import BB, BH, HB, _to_reserve, _with  # noqa: E402
from jepagen.limiter import RenderLimiter  # noqa: E402
from jepagen.run import run_episodes_batched  # noqa: E402
from jepagen.sampler import Manifest, Template, _place_in, _size_for, _volume, _z_extent, load_yaml, sample_episode  # noqa: E402
from jepagen.shards import clip_to_sample  # noqa: E402
from jepagen.spec import Transfer  # noqa: E402
from jepagen.validate import validate_clip  # noqa: E402
from reversal_items import _drop  # noqa: E402

SMALL, BIG, A_RANGE = (0.035, 0.040), (0.058, 0.065), (0.035, 0.065)
SHAPES = ("sphere", "box", "cylinder")


def base_ok(ep) -> bool:
    return (ep.violation is None and not ep.glides and sum(isinstance(a, Transfer) for a in ep.actions) == 2
            and len(ep.objects) >= 1)


def _area(shape, r):
    return r * r      # every dimension scales with r for every shape, so ratios are shape-free


def _vol(shape, r):
    return r ** 3


def build(base, seed: int, direction: str, kinds: tuple[str, ...]) -> dict[str, object]:
    """{f'{kind}:{hist}': Episode} for kind in kinds (conc / disc), hist in (two, one)."""
    rng = np.random.default_rng([seed, 61])
    shape = SHAPES[int(rng.integers(3))]
    bx, hp = base.scene.box, base.scene.hopper
    proto = sorted(base.objects, key=lambda o: o.id)
    colors = [proto[i % len(proto)].color for i in range(6)]
    if len(proto) < 6:      # extra colours: the base's own, hue-rotated, so every object stays distinct
        for i in range(len(proto), 6):
            c = proto[i % len(proto)].color
            colors[i] = (c[1], c[2], c[0], c[3])
    r_a = float(rng.uniform(*A_RANGE))
    sm = [float(rng.uniform(*SMALL)) for _ in range(3)]
    bg = [float(rng.uniform(*BIG)) for _ in range(2)]
    hints = [{"box": _drop(rng, bx), "hopper": _drop(rng, hp)} for _ in range(2)]

    def spots(tray, n):
        # positions safe for the largest radius, shared by every history of the scene
        return _place_in(rng, tray, [(shape, _size_for(shape, BIG[1]), BIG[1])] * n)

    def obj(i, r, region, pos):
        size = _size_for(shape, r)
        o = proto[0].model_copy(update={"id": i, "shape": shape, "size": size, "color": colors[i],
                                        "mass": float(_volume(shape, size) * 600), "start_region": "box"})
        if region == "reserve":
            return _to_reserve(o)
        z = tray_floor[region] + _z_extent(shape, size) + 0.004
        return o.model_copy(update={"start_region": region, "start_pos": (pos[0], pos[1], z),
                                    "start_quat": proto[min(i, len(proto) - 1)].start_quat})

    tray_floor = {"box": bx.wall_thickness, "hopper": hp.wall_thickness}
    out = {}
    if direction == "add":
        n0 = int(rng.integers(0, 2))
        bpos, hpos = spots(bx, 1), spots(hp, 2)
        one_first_hb = True if n0 == 0 else bool(rng.integers(2))
        for kind in kinds:
            # the claw takes the taller hopper object: in 'one' the carried Y must be the larger of Y and f
            X, Y = (bg, max(sm[0], sm[2])) if kind == "conc" else (sm[:2], bg[0])
            f = min(sm[0], sm[2]) if kind == "conc" else sm[2]
            for hist in ("two", "one"):
                objs = [obj(0, r_a, "box" if n0 else "reserve", bpos[0])]
                if hist == "two":
                    objs += [obj(1, X[0], "hopper", hpos[0]), obj(2, X[1], "hopper", hpos[1])]
                    routes, carried = (HB, HB), list(X)
                else:
                    objs += [obj(1, Y, "hopper", hpos[0]), obj(2, f, "hopper", hpos[1])]
                    routes, carried = ((HB, BB) if one_first_hb else (BB, HB)), [Y]
                objs.append(obj(3, SMALL[0], "reserve", None))
                radii = {0: r_a, 1: (X[0] if hist == "two" else Y), 2: (X[1] if hist == "two" else f), 3: SMALL[0]}
                final_r = ([r_a] if n0 else []) + carried
                meta = dict(base.meta, n0=n0, op="add", k=len(carried), final=len(final_r),
                            shuffles=sum(x == BB for x in routes), n_hopper=2, route=[list(x) for x in routes],
                            d1_kind=kind, d1_hist=hist, d1_dir="add", shape=shape, scene_seed=seed, radii=radii,
                            final_radii=final_r, final_classes=sorted(_cls(r) for r in final_r))
                out[f"{kind}:{hist}"] = _with(base, objs, routes, hints, meta, f"d1-{seed:09d}-{kind}-{hist}")
        if "disc" in kinds:
            two, one = out["disc:two"].meta["final_radii"], out["disc:one"].meta["final_radii"]
            if not (sum(_area(shape, r) for r in two) < sum(_area(shape, r) for r in one)
                    and sum(_vol(shape, r) for r in two) < sum(_vol(shape, r) for r in one)):
                raise ValueError("discordant pair is not discordant in area and volume")
    else:
        n0 = int(rng.choice([2, 3]))
        bpos, hpos = spots(bx, 3), spots(hp, 1)
        box_r = ([r_a] if n0 == 3 else []) + [sm[0], bg[0]]
        first_bh = bool(rng.integers(2))
        for hist in ("two", "one"):
            objs = [obj(i, r, "box", bpos[i]) for i, r in enumerate(box_r)]
            objs.append(obj(len(objs), sm[2], "hopper", hpos[0]))
            radii = dict(enumerate(box_r + [sm[2]]))
            while len(objs) < 4:
                radii[len(objs)] = SMALL[0]
                objs.append(obj(len(objs), SMALL[0], "reserve", None))
            routes = (BH, BH) if hist == "two" else ((BH, BB) if first_bh else (BB, BH))
            k = 2 if hist == "two" else 1
            # the claw takes the tallest first: big, then (for 'two') the taller of the rest
            rest = sorted(box_r, reverse=True)[k:]
            meta = dict(base.meta, n0=n0, op="remove", k=k, final=n0 - k, shuffles=sum(x == BB for x in routes),
                        n_hopper=1, route=[list(x) for x in routes], d1_kind="conc", d1_hist=hist, d1_dir="remove",
                        shape=shape, scene_seed=seed, final_radii=rest, final_classes=None, radii=radii)
            out[f"conc:{hist}"] = _with(base, objs, routes, hints, meta, f"d1-{seed:09d}-rconc-{hist}")
    return out


def _cls(r):
    return "big" if r >= BIG[0] - 1e-9 else "small" if r <= SMALL[1] + 1e-9 else "mid"


def final_box(clip) -> list[int]:
    last = clip.states[-1]
    return sorted(o["id"] for o in last.objects if o["region"] == "box")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=("A", "B", "R"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=os.path.join(HERE, "..", "configs", "v4b_small.yaml"))
    ap.add_argument("--seeds", type=int, nargs=2, required=True)
    ap.add_argument("--n", type=int, required=True, help="scenes to keep")
    ap.add_argument("--max-render-fps", type=float, default=60.0)
    ap.add_argument("--batch", type=int, default=8, help="scenes per render batch")
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit(f"{a.out} is not empty: refusing to append")
    direction = "remove" if a.subset == "R" else "add"
    kinds = ("conc", "disc") if a.subset == "A" else ("conc",)
    t, man = load_yaml(a.template, Template), Manifest()
    os.makedirs(a.out, exist_ok=True)
    lim = RenderLimiter(os.path.join(a.out, ".limiter"), a.max_render_fps)
    writers = {}
    for kind in kinds:
        d = os.path.join(a.out, f"{a.subset}_{kind}", "train", "possible")
        os.makedirs(d, exist_ok=True)
        writers[kind] = wds.ShardWriter(os.path.join(d, "shard-%06d.tar"), maxcount=500, verbose=0)
    log = open(os.path.join(a.out, "items.jsonl"), "a")
    rej = open(os.path.join(a.out, "rejects.jsonl"), "a")
    kept, pending = 0, []

    def flush():
        nonlocal kept
        res = run_episodes_batched([e for _, h in pending for e in h.values()], lim)
        j = 0
        for seed, h in pending:
            reasons, clips = [], {}
            for k, ep in h.items():
                clip, diag = res[j]; j += 1
                clips[k] = (clip, diag)
                v = validate_clip(clip, diag)
                reasons += [f"{k}: {r}" for r in v.reasons]
                g = [e for e in clip.events if e["event"] == "grasp"]
                rl = [e for e in clip.events if e["event"] == "release"]
                if len(g) != 2 or len(rl) != 2:
                    reasons.append(f"{k}: {len(g)} grasps / {len(rl)} releases")
                    continue
                fb = final_box(clip)
                radii = ep.meta["radii"]
                got = sorted(_cls(radii[i]) for i in fb)
                if len(fb) != ep.meta["final"]:
                    reasons.append(f"{k}: {len(fb)} objects in the box at the end, planned {ep.meta['final']}")
                elif ep.meta["final_classes"] is not None and got != ep.meta["final_classes"]:
                    reasons.append(f"{k}: final box classes {got}, planned {ep.meta['final_classes']}")
                elif ep.meta["d1_dir"] == "remove" and sorted(radii[i] for i in fb) != sorted(ep.meta["final_radii"]):
                    reasons.append(f"{k}: removal left {sorted(radii[i] for i in fb)}, planned {ep.meta['final_radii']}")
            if reasons or kept >= a.n:
                if reasons:
                    rej.write(json.dumps({"seed": seed, "reasons": reasons}) + "\n"); rej.flush()
                continue
            for k, (clip, diag) in clips.items():
                writers[k.split(":")[0]].write(clip_to_sample(clip, "rust", validate_clip(clip, diag).metrics))
            log.write(json.dumps({"seed": seed, "names": list(h), "meta": {k: h[k].meta for k in h}}, default=float) + "\n")
            log.flush()
            kept += 1
        pending.clear()

    for seed in range(*a.seeds):
        if kept >= a.n:
            break
        base = sample_episode(t, man, "train", seed)
        if not base_ok(base):
            continue
        try:
            h = build(base, seed, direction, kinds)
        except (ValueError, RuntimeError) as ex:
            rej.write(json.dumps({"seed": seed, "reasons": [f"build: {ex}"]}) + "\n"); rej.flush()
            continue
        pending.append((seed, h))
        if len(pending) == a.batch:
            flush()
            print(f"seed {seed}: kept {kept}", flush=True)
    if pending:
        flush()
    for w in writers.values():
        w.close()
    print(json.dumps({"kept": kept}))
    if kept < a.n:
        raise SystemExit(f"only {kept} of {a.n} scenes kept: widen --seeds")


if __name__ == "__main__":
    main()
