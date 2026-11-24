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


def token_seg(seg: np.ndarray, res: int, patch: int, tubelet: int, min_frac: float = 0.125) -> np.ndarray:
    """Per-token object labels for the mask branch: seg [T, H, W] (0 = background, 1..250 = object id + 1,
    253/254 = screen/claw) -> uint8 [T // tubelet, G, G], G = res // patch, slot-major like the encoder's tokens.
    A token's label is the object covering the most of its tubelet volume, if objects cover at least min_frac
    of it; else 0 (no object)."""
    T, H, W = seg.shape
    G = res // patch
    if H != W or H % G or T % tubelet:
        raise ValueError(f"seg {seg.shape} does not tile into {T // tubelet} x {G} x {G} tokens")
    b = H // G
    blk = seg.reshape(T // tubelet, tubelet, G, b, G, b).transpose(0, 2, 4, 1, 3, 5).reshape(T // tubelet, G, G, -1)
    obj = (blk >= 1) & (blk <= 250)
    ids = np.unique(blk[obj])
    out = np.zeros((T // tubelet, G, G), dtype=np.uint8)
    if len(ids):
        counts = np.stack([(blk == k).sum(-1) for k in ids])            # [K, S, G, G]
        best = counts.argmax(0)
        enough = obj.sum(-1) >= min_frac * blk.shape[-1]
        out[enough] = ids[best][enough]
    return out


def motion_seg(frames: np.ndarray, patch: int, tubelet: int, thresh: float = 0.06, min_area: int = 12,
               link_iou: float = 0.2, min_frac: float = 0.125, cue: str = "chroma") -> np.ndarray:
    """Label-free object groups from motion (common fate), in the layout of token_seg: frames uint8 [T, H, W, 3]
    -> uint8 [T // tubelet, G, G], G = H // patch, 0 = no group. Uses pixels only, no renderer labels:
    background = per-pixel temporal median; foreground = pixels differing from it by > thresh, cleaned by a 3x3
    opening. cue "chroma" compares brightness-normalised colour (r, g, b) / (r + g + b), so moving shadows (a
    brightness change) are not foreground; cue "rgb" compares raw values / 255 (max over channels); groups = connected foreground blobs of >= min_area pixels in each
    tubelet's middle frame, linked across tubelets by mask IoU >= link_iou (greedy, so a group keeps one id
    over time). A token takes the group covering most of its patch if groups cover >= min_frac of it.
    Known errors, kept on purpose (they are what a real grouper gets): objects that never move are not
    grouped; touching objects (and an object with the claw carrying it) merge; an object resting at one
    place for most of the clip joins the background, so its other positions show up as ghost blobs."""
    from scipy import ndimage
    T, H, W, _ = frames.shape
    G = H // patch
    if H != W or H % patch or T % tubelet:
        raise ValueError(f"frames {frames.shape} do not tile into {T // tubelet} x {G} x {G} tokens")
    if cue == "chroma":
        x = frames.astype(np.float32)
        x = x / (x.sum(-1, keepdims=True) + 30.0)      # + 30: dark pixels have no reliable chroma
    elif cue == "rgb":
        x = frames.astype(np.float32) / 255.0
    else:
        raise ValueError(f"unknown cue {cue!r}")
    bg = np.median(x, axis=0)
    out = np.zeros((T // tubelet, G, G), dtype=np.uint8)
    prev: list[tuple[int, np.ndarray]] = []           # (group id, mask) in the previous tubelet
    next_id = 1
    for s in range(T // tubelet):
        fg = np.abs(x[s * tubelet + tubelet // 2] - bg).max(-1) > thresh
        fg = ndimage.binary_opening(fg, structure=np.ones((3, 3), bool))
        lab, n = ndimage.label(fg)
        cur: list[tuple[int, np.ndarray]] = []
        taken: set[int] = set()
        for c in range(1, n + 1):
            m = lab == c
            if m.sum() < min_area:
                continue
            best, best_iou = None, link_iou
            for gid, pm in prev:
                if gid in taken:
                    continue
                iou = (m & pm).sum() / (m | pm).sum()
                if iou >= best_iou:
                    best, best_iou = gid, iou
            if best is None:
                if next_id > 255:
                    raise ValueError("motion_seg: more than 255 groups in one clip")
                best, next_id = next_id, next_id + 1
            taken.add(best)
            cur.append((best, m))
        grid = np.zeros((H, W), dtype=np.int32)
        for gid, m in cur:
            grid[m] = gid
        blk = grid.reshape(G, patch, G, patch).transpose(0, 2, 1, 3).reshape(G, G, -1)
        for gid, _m in cur:
            cnt = (blk == gid).sum(-1)
            sel = (cnt >= min_frac * blk.shape[-1]) & (cnt > ((blk > 0) & (blk != gid)).sum(-1))
            out[s][sel] = gid
        prev = cur
    return out


class PossibleClips(torch.utils.data.IterableDataset):
    """Endless stream of possible clips (uint8 [T, res, res, 3]) from v2 shards, minus held-out seeds.
    Shards are split across workers and reshuffled every pass; a buffer shuffles clips within a worker."""

    def __init__(self, data: str, exclude_ids: set[str], res: int, split: str = "train", buffer: int = 256,
                 seed: int = 0, tokseg: bool = False):
        self.shards = sorted(glob.glob(os.path.join(data, split, "possible", "*.tar")))
        if not self.shards:
            raise FileNotFoundError(f"no possible shards under {data}/{split}")
        self.exclude, self.buffer, self.seed, self.res = exclude_ids, buffer, seed, res
        self.tokseg = tokseg  # also yield per-token object labels (shards from prep_cloud --tokseg)

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
                clip = torch.from_numpy(downsample(frames, self.res))
                if self.tokseg:
                    if "tokseg.npy" not in s:
                        raise KeyError(f"{s['__key__']}: no tokseg.npy (prepare shards with prep_cloud --tokseg)")
                    clip = (clip, torch.from_numpy(np.load(io.BytesIO(s["tokseg.npy"]))))
                buf.append(clip)
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
