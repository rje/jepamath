"""Training clips (possible only) and the held-out evaluation set for Track B."""

from __future__ import annotations

import glob
import io
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds

FRAME_KEYS = ("possible", "impossible", "control")


def downsample(frames: np.ndarray, res: int) -> np.ndarray:
    """uint8 [T, H, W, 3] -> uint8 [T, res, res, 3], antialiased bilinear (as jepaeval preprocess does)."""
    if frames.shape[1] == res and frames.shape[2] == res:
        return frames
    if frames.shape[1] != frames.shape[2]:
        raise ValueError(f"non-square frames {frames.shape}")
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(res, res), mode="bilinear", antialias=True, align_corners=False)
    return x.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()


def pack(frames: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, frames=frames)
    return buf.getvalue()


class PackedTriple(dict):
    """A held-out triple whose clips are stored as compressed npz bytes and decoded on each access.
    2,000 triples at 256 px are ~90 GB raw, too much to hold in RAM; the flat renders compress well."""

    def __getitem__(self, k):
        v = super().__getitem__(k)
        return np.load(io.BytesIO(v))["frames"] if k in FRAME_KEYS else v


def heldout_res(triples: list[dict]) -> int:
    h, w = triples[0]["possible"].shape[1:3]
    if h != w:
        raise ValueError(f"non-square held-out frames {h}x{w}")
    return int(h)


def build_heldout(data: str, out: str, n: int, res: int, split: str = "train") -> None:
    """First n twin triples (the same seeds jepaeval's reveal eval uses), at `res` px, clips compressed
    (PackedTriple), saved for reuse. Their seeds are excluded from training."""
    from jepaeval.twins import load_triples
    if PackedTriple.__module__ != "jepatrain.data":
        raise RuntimeError(f"PackedTriple would pickle as {PackedTriple.__module__}.PackedTriple; "
                           "call jepatrain.data.build_heldout from the imported module")
    triples = [PackedTriple(t) for t in load_triples(data, split, n, transform=lambda f: pack(downsample(f, res)))]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(triples, out)
    with open(out + ".ids.json", "w") as f:
        json.dump([t["seed_id"] for t in triples], f)


class PossibleClips(torch.utils.data.IterableDataset):
    """Endless stream of possible clips (uint8 [T, res, res, 3]) from v2 shards, minus held-out seeds.
    Shards are split across workers and reshuffled every pass; a buffer shuffles clips within a worker."""

    def __init__(self, data: str, exclude_ids: set[str], res: int, split: str = "train", buffer: int = 256,
                 seed: int = 0):
        self.shards = sorted(glob.glob(os.path.join(data, split, "possible", "*.tar")))
        if not self.shards:
            raise FileNotFoundError(f"no possible shards under {data}/{split}")
        self.exclude, self.buffer, self.seed, self.res = exclude_ids, buffer, seed, res

    def __iter__(self):
        torch.set_num_threads(1)
        wi = torch.utils.data.get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = random.Random(self.seed * 1000 + wid)
        mine = self.shards[wid::nw]
        if not mine:
            raise RuntimeError(f"worker {wid} has no shards ({len(self.shards)} shards, {nw} workers)")
        buf = []
        while True:
            rng.shuffle(mine)
            seen = 0
            # this class already splits shards across workers; WebDataset's own worker split would split again
            for s in wds.WebDataset(mine, shardshuffle=False, empty_check=False, workersplitter=lambda src: src):
                seen += 1
                if s["__key__"] in self.exclude:
                    continue
                frames = np.load(io.BytesIO(s["clip.npz"]))["frames"]
                buf.append(torch.from_numpy(downsample(frames, self.res)))
                if len(buf) >= self.buffer:
                    yield buf.pop(rng.randrange(len(buf)))
            if seen == 0:
                raise RuntimeError(f"worker {wid}: a full pass over {mine} yielded no samples")


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="jepatrain.data", description="build the held-out triple set")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--res", type=int, required=True)
    a = ap.parse_args()
    # run as `python -m jepatrain.data` this file is __main__; build through the importable module so the
    # pickle names jepatrain.data.PackedTriple, which any other process can load
    from jepatrain.data import build_heldout as build
    build(a.data, a.out, a.n, a.res)


if __name__ == "__main__":
    main()
