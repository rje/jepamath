"""History-reversal items (docs/2026-11-26-history-reversal-design.md): per scene, four natural histories with the
same start and the same two-transfer timeline. Only each transfer's source and destination differ, so the final
box counts are four distinct values in 0-3. Nothing hidden is changed. Each history's prediction is later scored
against every history's ending.

  n0 = 1: H++ (3), H+ (2), H0 (1), H- (0)
  n0 = 2: H+ (3), H0 (2), H- (1), H-- (0)

Single-op histories put their shuffle in slot 1 or slot 2 by a per-scene coin, so neither slot alone gives the
magnitude. The exception is n0 = 1 H-, whose shuffle must come first because the box is empty after the removal.

Every clip passes validate_clip, which checks counts, occlusion leaks and that each carried object shows. On top
of that, each item is checked for:
  - the four histories being pixel-identical before transfer 1 starts;
  - every grasped object being visible while held in a frame before the screen starts lowering (model context);
  - at least one tubelet whose first frame is at or after the reveal end.
A scene is kept only if all four of its clips pass.

    python scripts/reversal_items.py --out DIR --seeds 900000 900400 --per-n0 20
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
from jepagen.limiter import RenderLimiter  # noqa: E402
from jepagen.run import run_episodes_batched  # noqa: E402
from jepagen.sampler import Manifest, Template, _project, load_yaml, sample_episode  # noqa: E402
from jepagen.spec import Episode, ScreenMove, Transfer  # noqa: E402
from jepagen.validate import min_visible_px, validate_clip  # noqa: E402
from jepatrain.data import downsample  # noqa: E402

HB, BB, BH = ("hopper", "box"), ("box", "box"), ("box", "hopper")
TUBELET, RES = 4, 128


def histories(n0: int, coin_p: bool, coin_m: bool) -> dict[str, tuple]:
    """History name -> (transfer 1 (src, dst), transfer 2 (src, dst)). coin_*: shuffle in slot 1 instead of 2."""
    plus = (BB, HB) if coin_p else (HB, BB)
    if n0 == 1:
        return {"H++": (HB, HB), "H+": plus, "H0": (BB, BB), "H-": (BB, BH)}
    if n0 == 2:
        minus = (BB, BH) if coin_m else (BH, BB)
        return {"H+": plus, "H0": (BB, BB), "H-": minus, "H--": (BH, BH)}
    raise ValueError(f"n0 {n0} not in (1, 2)")


def _drop(rng, tray) -> tuple[float, float]:
    """The sampler's drop hint (sampler.drop_xy), in the given tray."""
    m = 0.045
    return (float(tray.center[0] + rng.uniform(-tray.inner_size[0] / 2 + m, tray.inner_size[0] / 2 - m)),
            float(tray.center[1] + rng.uniform(-tray.inner_size[1] / 2 + m, tray.inner_size[1] / 2 - m)))


def base_ok(ep: Episode) -> bool:
    m = ep.meta
    return (m["visibility"] == "occluded" and m["op"] == "add" and m["k"] == 1 and m["shuffles"] == 1
            and m["n0"] in (1, 2) and m["n_hopper"] >= 2 and not ep.glides)


def make_histories(base: Episode, seed: int) -> dict[str, Episode]:
    """Four episodes from one base: same objects, scene, screen moves and transfer times; per-history
    src/dst, destination-relative drop hints (drawn once per slot and tray) and meta."""
    rng = np.random.default_rng([seed, 99])
    tr = [a for a in base.actions if isinstance(a, Transfer)]
    if len(tr) != 2:
        raise ValueError(f"{base.episode_id}: {len(tr)} transfers, need 2")
    hints = [{"box": _drop(rng, base.scene.box), "hopper": _drop(rng, base.scene.hopper)} for _ in tr]
    coin_p, coin_m = bool(rng.integers(2)), bool(rng.integers(2))
    n0, nh = base.meta["n0"], base.meta["n_hopper"]
    out = {}
    for name, route in histories(n0, coin_p, coin_m).items():
        acts, i = [], 0
        for a in base.actions:
            if isinstance(a, Transfer):
                src, dst = route[i]
                acts.append(a.model_copy(update={"src": src, "dst": dst, "drop_xy": hints[i][dst]}).model_dump())
                i += 1
            else:
                acts.append(a.model_dump())
        into = sum(r == HB for r in route)
        out_ = sum(r == BH for r in route)
        op, k = ("add", into) if into else (("remove", out_) if out_ else ("none", 0))
        meta = dict(base.meta, op=op, k=k, shuffles=sum(r == BB for r in route), final=n0 + into - out_,
                    history=name, route=[list(r) for r in route], scene_seed=seed)
        out[name] = Episode.model_validate({**base.model_dump(), "episode_id": f"rev-{seed:09d}-{name}",
                                            "actions": acts, "meta": meta})
    finals = sorted(e.meta["final"] for e in out.values())
    if finals != [0, 1, 2, 3]:
        raise AssertionError(f"finals {finals}")
    return out


def box_tokens(ep: Episode, grid: int = RES // 8) -> np.ndarray:
    """Bool [grid, grid]: patches overlapping the projected box interior volume (inner floor up to the rim plus
    the pile height), in model image coordinates (NDC y up -> image rows down)."""
    from matplotlib.path import Path
    from scipy.spatial import ConvexHull
    b, cam, c = ep.scene.box, ep.scene.camera, ep.clip
    hx, hy = b.inner_size[0] / 2, b.inner_size[1] / 2
    zs = (0.0, b.wall_height + 0.14)
    pts = np.array([(b.center[0] + sx * hx, b.center[1] + sy * hy, z) for sx in (-1, 1) for sy in (-1, 1) for z in zs])
    ndc = _project(cam, c.width / c.height, pts)
    px = np.stack([(ndc[:, 0] + 1) / 2 * RES, (1 - ndc[:, 1]) / 2 * RES], 1)
    hull = Path(px[ConvexHull(px).vertices])
    s = RES // grid
    yy, xx = np.mgrid[0:RES, 0:RES]
    inside = hull.contains_points(np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], 1)).reshape(RES, RES)
    frac = inside.reshape(grid, s, grid, s).mean((1, 3))
    return frac >= 0.25


def item_checks(clips: dict, fps: float) -> list[str]:
    """Cross-history and model-context checks (see module doc)."""
    reasons = []
    eps = {k: c.episode for k, c in clips.items()}
    e0 = next(iter(eps.values()))
    t1_start = min(a.t_start for a in e0.actions if isinstance(a, Transfer))
    lower = next(a for a in e0.actions if isinstance(a, ScreenMove) and a.kind == "screen_lower")
    times = e0.clip.frame_times()
    ref = next(iter(clips.values()))
    for k, c in clips.items():
        for i, t in enumerate(times):
            if t >= t1_start:
                break
            if not np.array_equal(c.frames[i], ref.frames[i]):
                reasons.append(f"{k}: differs from the others before transfer 1 (t={t:.2f})")
                break
    # context frames, at the model's rate (the battery subsamples a 32-frame clip to its 32-frame window: step 1)
    ctx = int(lower.t_start * fps) // TUBELET * TUBELET
    if ctx < TUBELET:
        reasons.append("no context tubelet before the screen lowers")
    f_end = math.ceil(lower.t_end * fps)
    first = next((j for j in range((len(times) - ctx) // TUBELET) if ctx + TUBELET * j >= f_end), None)
    if first is None:
        reasons.append("no tubelet whose first frame is after the reveal")
    for k, c in clips.items():
        mv = min_visible_px(c.episode.clip.width, c.episode.clip.height)
        ev = c.events
        grasps = [(e["t"], e["object_id"]) for e in ev if e["event"] == "grasp"]
        rels = [e["t"] for e in ev if e["event"] == "release"]
        if len(grasps) != 2 or len(rels) != 2:
            reasons.append(f"{k}: {len(grasps)} grasps / {len(rels)} releases, need 2 / 2")
            continue
        for (tg, oid), tr in zip(grasps, rels):
            seen = [int((c.seg[i] == oid + 1).sum()) for i, t in enumerate(times) if tg <= t <= tr and i < ctx]
            if not seen or max(seen) < mv:
                reasons.append(f"{k}: object {oid} carried {tg:.2f}-{tr:.2f}s not visible in context ({max(seen, default=0)} px)")
    return reasons, {"ctx": ctx, "first_post_slot": first}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=os.path.join(HERE, "..", "configs", "v4b_small.yaml"))
    ap.add_argument("--seeds", type=int, nargs=2, required=True)
    ap.add_argument("--per-n0", type=int, required=True, help="accepted scenes wanted for each n0")
    ap.add_argument("--max-render-fps", type=float, default=60.0)
    ap.add_argument("--batch", type=int, default=16, help="scenes per GPU submission (4 clips each)")
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit(f"{a.out} is not empty: refusing to append (duplicate scenes would narrow the bootstrap)")
    os.makedirs(a.out, exist_ok=True)
    t, man = load_yaml(a.template, Template), Manifest()
    lim = RenderLimiter(os.path.join(a.out, ".limiter"), a.max_render_fps)
    kept = {1: 0, 2: 0}
    log = open(os.path.join(a.out, "items.jsonl"), "a")
    rej = open(os.path.join(a.out, "rejects.jsonl"), "a")
    pending = []

    def flush():
        eps = [e for _, h in pending for e in h.values()]
        res = run_episodes_batched(eps, lim)
        j = 0
        for seed, h in pending:
            clips, reasons = {}, []
            for k in h:
                clip, diag = res[j]; j += 1
                v = validate_clip(clip, diag)
                if not v.ok:
                    reasons += [f"{k}: {r}" for r in v.reasons]
                clips[k] = clip
            r2, info = item_checks(clips, h[next(iter(h))].clip.fps)
            reasons += r2
            n0 = h[next(iter(h))].meta["n0"]
            if reasons or kept[n0] >= a.per_n0:
                if reasons:
                    rej.write(json.dumps({"seed": seed, "n0": n0, "reasons": reasons}) + "\n"); rej.flush()
                continue
            names = sorted(h, key=lambda k: h[k].meta["final"])
            np.savez_compressed(os.path.join(a.out, f"scene_{seed:09d}.npz"),
                                frames=np.stack([downsample(clips[k].frames, RES) for k in names]),
                                box_tokens=box_tokens(h[names[0]]))
            log.write(json.dumps({"seed": seed, "n0": n0, "histories": names, **info,
                                  "meta": {k: h[k].meta for k in names},
                                  "episodes": {k: h[k].model_dump() for k in names}}, default=float) + "\n")
            log.flush()
            kept[n0] += 1
        pending.clear()

    for seed in range(*a.seeds):
        if all(v >= a.per_n0 for v in kept.values()):
            break
        base = sample_episode(t, man, "train", seed)  # train split rules; seed range disjoint from all v4b sets
        if not base_ok(base) or kept[base.meta["n0"]] >= a.per_n0:
            continue
        pending.append((seed, make_histories(base, seed)))
        if len(pending) == a.batch:
            flush()
            print(f"seed {seed}: kept {kept}", flush=True)
    if pending:
        flush()
    print(json.dumps({"kept": kept}))
    if any(v < a.per_n0 for v in kept.values()):
        raise SystemExit(f"seed range exhausted before {a.per_n0} scenes per n0: {kept}")


if __name__ == "__main__":
    main()
