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
from .sampler import Manifest, Template, make_twin, sample_episode
from .shards import Shards, check_disk, clip_to_sample, index_row
from .validate import validate_clip, validate_twins


@dataclass(frozen=True)
class Job:
    template: dict
    manifest: dict
    split: str
    backend: str
    twins: bool
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


def produce(job: Job, seed: int) -> dict:
    """Runs one seed. Returns accepted samples, or the rejection reasons. Never raises for a bad
    clip (that is a rejection, recorded); raises for anything else."""
    t, m = Template.model_validate(job.template), Manifest.model_validate(job.manifest)
    lim = _limiter(job)
    ep = sample_episode(t, m, job.split, seed)
    clip, diag = run_episode(ep, job.backend, lim)
    val = validate_clip(clip, diag)
    out = {"seed": seed, "episode_id": ep.episode_id, "accepted": [], "rejected": []}
    if not val.ok:
        out["rejected"].append({"episode_id": ep.episode_id, "reasons": val.reasons})
        return out
    items = [(clip, val)]
    if job.twins:
        tw = make_twin(ep, seed)
        tclip, tdiag = run_episode(tw, job.backend, lim)
        tval = validate_clip(tclip, tdiag)
        pval = validate_twins(clip, tclip)
        if not (tval.ok and pval.ok):
            # the pair is the unit: drop both so every impossible clip has its possible twin
            out["rejected"].append({"episode_id": tw.episode_id, "reasons": tval.reasons + pval.reasons})
            return out
        items.append((tclip, tval))
    for c, v in items:
        out["accepted"].append((c, clip_to_sample(c, job.backend, v.metrics)))
    return out


def _judge(job: Job, seed: int, ep, clip, diag, tw=None, tclip=None, tdiag=None) -> dict:
    val = validate_clip(clip, diag)
    out = {"seed": seed, "episode_id": ep.episode_id, "accepted": [], "rejected": []}
    if not val.ok:
        out["rejected"].append({"episode_id": ep.episode_id, "reasons": val.reasons})
        return out
    items = [(clip, val)]
    if tw is not None:
        tval = validate_clip(tclip, tdiag)
        pval = validate_twins(clip, tclip)
        if not (tval.ok and pval.ok):
            out["rejected"].append({"episode_id": tw.episode_id, "reasons": tval.reasons + pval.reasons})
            return out
        items.append((tclip, tval))
    for c, v in items:
        out["accepted"].append((c, clip_to_sample(c, job.backend, v.metrics)))
    return out


def produce_batch(job: Job, seeds: list[int]) -> list[dict]:
    """Runs several seeds (and their twins) as one batched render: every frame of every clip in
    the group is one GPU submission. Same outputs as calling `produce` per seed."""
    t, m = Template.model_validate(job.template), Manifest.model_validate(job.manifest)
    lim = _limiter(job)
    eps = [sample_episode(t, m, job.split, s) for s in seeds]
    tws = [make_twin(ep, s) for ep, s in zip(eps, seeds)] if job.twins else []
    res = run_episodes_batched(eps + tws, lim)
    outs = []
    for j, (s, ep) in enumerate(zip(seeds, eps)):
        clip, diag = res[j]
        if job.twins:
            tclip, tdiag = res[len(eps) + j]
            outs.append(_judge(job, s, ep, clip, diag, tws[j], tclip, tdiag))
        else:
            outs.append(_judge(job, s, ep, clip, diag))
    return outs


def _work(args):
    job, seeds = args
    if job.batch > 1:
        return produce_batch(job, seeds)
    return [produce(job, s) for s in seeds]


def generate(template: Template, manifest: Manifest, split: str, seeds: range, out_dir: str, backend: str = "mujoco",
             twins: bool = False, workers: int = 4, max_render_fps: float = 60.0, temp_ceiling_c: float = 83.0,
             batch: int = 1, est_bytes_per_clip: int = 400_000) -> dict:
    if batch < 1:
        raise ValueError("batch must be >= 1")
    if batch > 1 and backend != "rust":
        raise ValueError("batched rendering needs the rust backend")
    scenes_per_seed = 2 if twins else 1
    if batch % scenes_per_seed:
        raise ValueError(f"batch {batch} must be a multiple of {scenes_per_seed} (each seed renders {scenes_per_seed} clips)")
    seeds_per_task = max(1, batch // scenes_per_seed)
    os.makedirs(out_dir, exist_ok=True)
    n_clips = len(seeds) * (2 if twins else 1)
    check_disk(out_dir, n_clips * est_bytes_per_clip)
    job = Job(template.model_dump(mode="json"), manifest.model_dump(mode="json"), split, backend, twins,
              os.path.join(out_dir, ".render_limiter"), max_render_fps, temp_ceiling_c, batch)
    with open(os.path.join(out_dir, f"manifest-{split}.json"), "w") as f:
        json.dump({"template": job.template, "manifest": job.manifest, "split": split, "seeds": [seeds.start, seeds.stop],
                   "backend": backend, "twins": twins, "max_render_fps": max_render_fps, "batch": batch}, f, indent=1)

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
                for clip, sample in res["accepted"]:
                    d = shards.write(clip, sample)
                    rows.append(index_row(clip, backend, d))
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
