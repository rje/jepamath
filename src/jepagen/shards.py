"""Clip serialization and WebDataset shard writing."""

from __future__ import annotations

import dataclasses
import io
import json
import os
import shutil

import numpy as np
import webdataset as wds

from .backends.base import ClipResult

MIN_FREE_BYTES = 50 * 1024 ** 3


def clip_to_sample(clip: ClipResult, backend: str, validation_metrics: dict) -> dict:
    buf = io.BytesIO()
    np.savez_compressed(buf, frames=clip.frames, seg=clip.seg)
    state = {"backend": backend, "events": clip.events, "metrics": validation_metrics,
             "frames": [dataclasses.asdict(s) for s in clip.states]}
    return {"__key__": clip.episode.episode_id, "clip.npz": buf.getvalue(),
            "state.json": json.dumps(state).encode(), "episode.json": clip.episode.to_json().encode()}


def shard_dir(clip: ClipResult) -> str:
    return os.path.join(clip.episode.split, clip.episode.twin_kind)


def index_row(clip: ClipResult, backend: str, shard_dir: str) -> dict:
    ep, m = clip.episode, clip.episode.meta
    return {"episode_id": ep.episode_id, "seed": ep.seed, "split": ep.split, "backend": backend,
            "possible": ep.violation is None, "twin_kind": ep.twin_kind, "twin_of": ep.twin_of or "",
            "violation": ep.violation.kind if ep.violation else "",
            "n0": m["n0"], "op": m["op"], "k": m["k"], "final": m["final"],
            "shown_final": m.get("shown_final", m["final"]), "control_mode": m["control_mode"],
            "shuffles": m["shuffles"], "palette": m["palette"], "sampler": m["sampler"],
            "shard_dir": shard_dir}


def load_sample(sample: dict) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    z = np.load(io.BytesIO(sample["clip.npz"]))
    return json.loads(sample["episode.json"]), z["frames"], z["seg"], json.loads(sample["state.json"])


def check_disk(path: str, needed_bytes: int) -> None:
    free = shutil.disk_usage(path).free
    if free - needed_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"not enough disk: {free / 1e9:.0f} GB free, run needs ~{needed_bytes / 1e9:.0f} GB, "
                           f"and {MIN_FREE_BYTES / 1e9:.0f} GB must stay free")


class Shards:
    """One ShardWriter per (split, possible|impossible|control) directory, so twins can never
    land in a training shard."""

    def __init__(self, out_dir: str, max_bytes: int = 1_000_000_000):
        self.out_dir = out_dir
        self.max_bytes = max_bytes
        self.writers: dict[str, wds.ShardWriter] = {}

    def write(self, d: str, sample: dict) -> None:
        """d: shard subdirectory from shard_dir(); passed in so workers never ship raw clips here."""
        if d not in self.writers:
            full = os.path.join(self.out_dir, d)
            os.makedirs(full, exist_ok=True)
            existing = len([f for f in os.listdir(full) if f.endswith(".tar")])
            self.writers[d] = wds.ShardWriter(os.path.join(full, "shard-%06d.tar"), maxsize=self.max_bytes,
                                              maxcount=100_000, start_shard=existing, verbose=0)
        self.writers[d].write(sample)

    def close(self):
        for w in self.writers.values():
            w.close()
