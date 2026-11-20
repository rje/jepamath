"""Parallel generation: sample -> run -> validate -> (twin) -> shards + index."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq

from .limiter import RenderLimiter, gpu_status
from .run import run_episode, run_episodes_batched
from .sampler import Manifest, Template, can_control, make_twin, sample_episode
from .shards import Shards, check_disk, clip_to_sample, index_row, shard_dir
from .validate import validate_clip, validate_twins


@dataclass(frozen=True)
class Job:
    template: dict
    manifest: dict
    split: str
    backend: str
    twins: tuple[str, ...]  # twin kinds per seed: "impossible" (count changes), "control" (move/swap)
    limiter_path: str
    max_render_fps: float
    temp_ceiling_c: float
    batch: int  # scenes per GPU submission (rust only); 1 = unbatched


_LIMITER: RenderLimiter | None = None


def _limiter(job: Job) -> RenderLimiter:
    global _LIMITER
    if _LIMITER is None:
        _LIMITER = RenderLimiter(job.limiter_path, job.max_render_fps, temp_ceiling_c=job.temp_ceiling_c)
    return _LIMITER


def _episodes_for(job: Job, t: Template, m: Manifest, seed: int) -> list:
    """The possible episode followed by its twins. Control twins are skipped when the box ends empty
    (nothing to move or swap); that is the only case a requested twin is not made."""
    ep = sample_episode(t, m, job.split, seed)
    out = [ep]
    for kind in job.twins:
        if kind == "control" and not can_control(ep):
            continue
        out.append(make_twin(ep, seed, kind))
    return out


def _judge(job: Job, seed: int, eps: list, results: list) -> dict:
    """The seed's clips are one unit: if the possible clip or any twin fails, none are kept, so every
    twin in the dataset has its possible clip."""
    ep = eps[0]
    out = {"seed": seed, "episode_id": ep.episode_id, "accepted": [], "rejected": []}
    clip, diag = results[0]
    val = validate_clip(clip, diag)
    if not val.ok:
        out["rejected"].append({"episode_id": ep.episode_id, "reasons": val.reasons})
        return out
    items = [(clip, val.metrics)]
    for tw, (tclip, tdiag) in zip(eps[1:], results[1:]):
        tval = validate_clip(tclip, tdiag)
        pval = validate_twins(clip, tclip)
        if not (tval.ok and pval.ok):
            out["rejected"].append({"episode_id": tw.episode_id, "reasons": tval.reasons + pval.reasons})
            return out
        items.append((tclip, {**tval.metrics, **pval.metrics}))
    for c, metrics in items:
        # only the compressed sample and its index row go back to the parent, never raw frames
        d = shard_dir(c)
        out["accepted"].append((d, clip_to_sample(c, job.backend, metrics), index_row(c, job.backend, d)))
    return out


def produce_group(job: Job, seeds: list[int]) -> list[dict]:
    """Runs several seeds and their twins. With job.batch > 1 every frame of every clip in the group
    is one GPU submission; the clips are identical either way."""
    t, m = Template.model_validate(job.template), Manifest.model_validate(job.manifest)
    lim = _limiter(job)
    groups = [_episodes_for(job, t, m, s) for s in seeds]
    flat = [e for g in groups for e in g]
    if job.batch > 1:
        res = run_episodes_batched(flat, lim)
    else:
        res = [run_episode(e, job.backend, lim) for e in flat]
    outs, i = [], 0
    for s, g in zip(seeds, groups):
        outs.append(_judge(job, s, g, res[i:i + len(g)]))
        i += len(g)
    return outs


def produce(job: Job, seed: int) -> dict:
    return produce_group(job, [seed])[0]


def _work(args):
    job, seeds = args
    return produce_group(job, seeds)


def generate(template: Template, manifest: Manifest, split: str, seeds: range, out_dir: str, backend: str = "mujoco",
             twins: tuple[str, ...] = (), workers: int = 4, max_render_fps: float = 60.0, temp_ceiling_c: float = 83.0,
             batch: int = 1, est_bytes_per_clip: int = 400_000) -> dict:
    if batch < 1:
        raise ValueError("batch must be >= 1")
    if batch > 1 and backend != "rust":
        raise ValueError("batched rendering needs the rust backend")
    for k in twins:
        if k not in ("impossible", "control"):
            raise ValueError(f"unknown twin kind {k!r}")
    scenes_per_seed = 1 + len(twins)   # at most; control twins are skipped for empty boxes
    if batch > 1 and batch < scenes_per_seed:
        raise ValueError(f"batch {batch} is smaller than the {scenes_per_seed} clips one seed renders")
    seeds_per_task = max(1, batch // scenes_per_seed)
    os.makedirs(out_dir, exist_ok=True)
    n_clips = len(seeds) * (1 + len(twins))
    check_disk(out_dir, n_clips * est_bytes_per_clip)
    job = Job(template.model_dump(mode="json"), manifest.model_dump(mode="json"), split, backend, tuple(twins),
              os.path.join(out_dir, ".render_limiter"), max_render_fps, temp_ceiling_c, batch)
    with open(os.path.join(out_dir, f"manifest-{split}.json"), "w") as f:
        json.dump({"template": job.template, "manifest": job.manifest, "split": split, "seeds": [seeds.start, seeds.stop],
                   "backend": backend, "twins": list(twins), "max_render_fps": max_render_fps, "batch": batch}, f, indent=1)

    shards = Shards(out_dir)
    rows, n_rej = [], 0
    rej_path = os.path.join(out_dir, f"rejects-{split}.jsonl")
    t0 = last_log = time.monotonic()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool, open(rej_path, "a") as rej:
        seed_list = list(seeds)
        tasks = [(job, seed_list[k:k + seeds_per_task]) for k in range(0, len(seed_list), seeds_per_task)]
        done = 0
        for group in pool.imap(_work, tasks, chunksize=1):
            for res in group:
                done += 1
                for r in res["rejected"]:
                    rej.write(json.dumps(r) + "\n")
                    n_rej += 1
                for d, sample, row in res["accepted"]:
                    shards.write(d, sample)
                    rows.append(row)
            now = time.monotonic()
            if now - last_log > 30 or done == len(seeds):
                last_log = now
                rate = done / (now - t0)
                print(f"[gen] {done}/{len(seeds)} seeds, {len(rows)} clips, {n_rej} rejected, "
                      f"{rate:.2f} seeds/s, GPU {gpu_status()}", flush=True)
                check_disk(out_dir, (len(seeds) - done) * est_bytes_per_clip * scenes_per_seed)
    shards.close()
    idx_path = os.path.join(out_dir, f"index-{split}.parquet")
    table = pa.Table.from_pylist(rows)
    if os.path.exists(idx_path):
        table = pa.concat_tables([pq.read_table(idx_path), table])
    pq.write_table(table, idx_path)
    return {"clips": len(rows), "rejected": n_rej, "seconds": time.monotonic() - t0}
