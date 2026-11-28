"""Indexed clip and prefix schedules for a paired visible-placement continuation."""

import hashlib
import io
import json
import os
import random
import tarfile
from pathlib import Path

import numpy as np
import torch


def catalog(shards, exclude):
    entries = []
    for shard in sorted(shards):
        with tarfile.open(shard, "r:") as tar:
            for member in tar:
                if member.name.endswith(".clip.npz"):
                    name = member.name.removesuffix(".clip.npz")
                    if name not in exclude:
                        entries.append((name, shard.name, member.offset_data, member.size))
    return entries


def prepare(root, replay, exclude, steps=100000, batch=32):
    replay_entries = catalog((replay / "train/possible").glob("*.tar"), exclude)
    demo_entries = catalog((root / "demo").glob("*.tar"), set())
    seeds = json.loads((root / "demo/manifest.json").read_text())["seeds"]
    assert [e[0] for e in demo_entries] == [f"{s}-{j}" for s in seeds for j in range(4)]
    if len(replay_entries) < 1000 or len({e[0] for e in replay_entries}) != len(replay_entries):
        raise ValueError("incomplete replay index")
    (root / "catalog.json").write_text(json.dumps([replay_entries, demo_entries], separators=(",", ":")))
    rng = np.random.default_rng(970112)
    # Each adjacent pair contains one replay batch and one demonstration batch.
    demo = np.concatenate([np.array([False, True])[rng.permutation(2)] for _ in range(steps // 2)])
    slots = rng.integers(1, 8, steps, dtype=np.int8)
    slots[rng.random(steps) >= 0.8] = 0  # original short/long spatial masks
    timing = np.zeros(steps, dtype=np.int8)
    demo_ix = np.flatnonzero(demo)
    timing[demo_ix] = rng.permutation(np.resize(np.arange(3, 6), len(demo_ix)))
    treat = slots.copy()
    # Preserve the exact multiset in each 500-update block, including on resumption.
    for begin in range(0, steps, 500):
        ix = np.arange(begin, min(begin + 500, steps))
        ix = ix[demo[ix] & (slots[ix] >= 3) & (slots[ix] <= 5)]
        ordered = ix[np.argsort(timing[ix], kind="stable")]
        treat[ordered] = np.sort(slots[ix])
    ids = rng.integers(0, len(replay_entries), size=(steps, batch), dtype=np.int32)
    for i in demo_ix:
        ids[i] = 4 * rng.integers(0, len(seeds), size=batch) + rng.permutation(np.arange(batch) % 4)
    np.savez(root / "schedule.npz", ids=ids, timing=timing, control=slots, treatment=treat)
    assert np.array_equal(np.bincount(slots, minlength=8), np.bincount(treat, minlength=8))
    assert np.array_equal(treat[~demo], slots[~demo])
    summary = {
        "steps": steps,
        "batch": batch,
        "demo_scenes": len(seeds),
        "replay_clips": len(replay_entries),
        "demo_batches": len(demo_ix),
        "mask_histogram": np.bincount(slots, minlength=8).tolist(),
        "control_midcarry_batches": int(np.sum(demo & (slots == timing))),
        "treatment_midcarry_batches": int(np.sum(demo & (treat == timing))),
        "schedule_sha256": hashlib.sha256((root / "schedule.npz").read_bytes()).hexdigest(),
    }
    if summary["treatment_midcarry_batches"] <= 2 * summary["control_midcarry_batches"]:
        raise ValueError("insufficient timing contrast")
    (root / "design.json").write_text(json.dumps(summary, indent=2))
    return summary


class PlacementBatches(torch.utils.data.Dataset):
    def __init__(self, root, replay, arm, start, steps, batch):
        root = Path(root)
        self.directories = [Path(replay) / "train/possible", root / "demo"]
        self.entries = json.loads((root / "catalog.json").read_text())
        with np.load(root / "schedule.npz") as z:
            self.ids = z["ids"]
            self.timing = z["timing"]
            self.cutoffs = z[arm]
        if self.ids.shape != (steps, batch):
            raise ValueError("training dimensions differ from frozen schedule")
        self.start = start
        self.handles = {}

    def __len__(self):
        return len(self.ids) - self.start

    def __getitem__(self, index):
        step = self.start + index
        timing = int(self.timing[step])
        kind = int(timing != 0)
        clips = []
        for i in self.ids[step]:
            name, shard, offset, size = self.entries[kind][i]
            path = self.directories[kind] / shard
            if path not in self.handles:
                self.handles[path] = open(path, "rb", buffering=0)
            compressed = os.pread(self.handles[path].fileno(), size, offset)
            with np.load(io.BytesIO(compressed)) as z:
                clip = z["frames"]
            if timing > 3:
                delay = (timing - 3) * 4
                clip = np.concatenate([np.repeat(clip[:1], delay, axis=0), clip[:-delay]])
            clips.append(clip)
        return torch.from_numpy(np.stack(clips))


def placement_mask(cutoff, seed, step, slots, grid):
    from .train import sample_masks

    if cutoff:
        end = int(cutoff) * grid * grid
        return "prefix", torch.arange(end), torch.arange(end, slots * grid * grid)
    return sample_masks(random.Random((seed + 1) * 1000003 + step), slots, grid, 0.0)
