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
from .run import run_episode
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


def _work(args):
    job, seed = args
    return produce(job, seed)


def generate(template: Template, manifest: Manifest, split: str, seeds: range, out_dir: str, backend: str = "mujoco",
             twins: bool = False, workers: int = 4, max_render_fps: float = 30.0, temp_ceiling_c: float = 83.0,
             est_bytes_per_clip: int = 400_000) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    n_clips = len(seeds) * (2 if twins else 1)
    check_disk(out_dir, n_clips * est_bytes_per_clip)
    job = Job(template.model_dump(mode="json"), manifest.model_dump(mode="json"), split, backend, twins,
              os.path.join(out_dir, ".render_limiter"), max_render_fps, temp_ceiling_c)
    with open(os.path.join(out_dir, f"manifest-{split}.json"), "w") as f:
        json.dump({"template": job.template, "manifest": job.manifest, "split": split, "seeds": [seeds.start, seeds.stop],
                   "backend": backend, "twins": twins, "max_render_fps": max_render_fps}, f, indent=1)

    shards = Shards(out_dir)
    rows, n_rej = [], 0
    rej_path = os.path.join(out_dir, f"rejects-{split}.jsonl")
    t0 = last_log = time.monotonic()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool, open(rej_path, "a") as rej:
        for i, res in enumerate(pool.imap(_work, [(job, s) for s in seeds], chunksize=1)):
            for r in res["rejected"]:
                rej.write(json.dumps(r) + "\n")
                n_rej += 1
            for clip, sample in res["accepted"]:
                d = shards.write(clip, sample)
                rows.append(index_row(clip, backend, d))
            now = time.monotonic()
            if now - last_log > 30 or i + 1 == len(seeds):
                last_log = now
                rate = (i + 1) / (now - t0)
                print(f"[gen] {i + 1}/{len(seeds)} seeds, {len(rows)} clips, {n_rej} rejected, "
                      f"{rate:.2f} seeds/s, GPU {gpu_status()}", flush=True)
                check_disk(out_dir, (len(seeds) - i - 1) * est_bytes_per_clip * (2 if twins else 1))
    shards.close()
    idx_path = os.path.join(out_dir, f"index-{split}.parquet")
    table = pa.Table.from_pylist(rows)
    if os.path.exists(idx_path):
        table = pa.concat_tables([pq.read_table(idx_path), table])
    pq.write_table(table, idx_path)
    return {"clips": len(rows), "rejected": n_rej, "seconds": time.monotonic() - t0}
