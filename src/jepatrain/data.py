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


def track_seg(frames: np.ndarray, patch: int, tubelet: int, thresh: float = 0.06, min_area: int = 12,
              link_iou: float = 0.2, min_frac: float = 0.125, app_tol: float = 0.5, reid_dist: float = 0.25,
              ring: int = 3, max_area: float | None = 450, max_elong: float | None = 3.0) -> np.ndarray:
    """Label-free object groups from motion plus persistence (S2 tracking grouper), same layout and cue as
    motion_seg (chroma, temporal-median background, blobs linked by mask IoU). Pixels only. Three additions, each
    the infant principle it stands for:
      ghosts      : a blob whose mean colour matches the ring of pixels around it (within app_tol * thresh, max
                    over channels) is background showing through where an object rested, not an object: dropped.
      persistence : a track with no blob this tubelet keeps its last mask and id if the pixels there still match
                    its appearance (mean colour within app_tol * thresh) and still stand out from their ring. So an
                    object that stops, and is absorbed into the median background, stays one group with one id.
      re-identity : a track that is lost (occluded, or gone) is remembered; a blob with no IoU link (reappearing,
                    or moved further than its own width since the last tubelet) takes the id of the nearest lost or
                    active track whose appearance matches, within reid_dist * H of where it was last seen.
      hand-sized  : a blob larger than max_area px (at 128 px; scaled by (H / 128)^2) or with a bounding box more
                    than max_elong times longer than wide is a surface, not a graspable object: dropped. In v5 this
                    is mostly the moving occluder screen, 45% of motion_seg's grouped tokens (thresholds read off
                    renderer statistics on 150 clips, 25 Nov: objects reach 380 px at the 99th percentile)
    Kept errors, as in motion_seg: objects that never move are not grouped; touching objects merge."""
    from scipy import ndimage
    T, H, W, _ = frames.shape
    G = H // patch
    if H != W or H % patch or T % tubelet:
        raise ValueError(f"frames {frames.shape} do not tile into {T // tubelet} x {G} x {G} tokens")
    x = frames.astype(np.float32)
    x = x / (x.sum(-1, keepdims=True) + 30.0)
    bg = np.median(x, axis=0)
    tol = app_tol * thresh
    out = np.zeros((T // tubelet, G, G), dtype=np.uint8)
    ring_k = np.ones((2 * ring + 1, 2 * ring + 1), bool)

    def contrast(f, m, fg):
        """max-channel |mean colour inside - mean colour of the surrounding ring| (ring excludes other foreground);
        None if the ring is too small to judge."""
        r = ndimage.binary_dilation(m, ring_k) & ~ndimage.binary_dilation(m) & ~fg
        if r.sum() < min_area:
            return None
        return float(np.abs(f[m].mean(0) - f[r].mean(0)).max())

    def centroid(m):
        return np.argwhere(m).mean(0)

    active: dict[int, tuple[np.ndarray, np.ndarray]] = {}    # id -> (mask, mean colour), seen last tubelet
    lost: dict[int, tuple[np.ndarray, np.ndarray]] = {}      # id -> (centroid, mean colour)
    next_id = 1
    for s in range(T // tubelet):
        f = x[s * tubelet + tubelet // 2]
        fg = np.abs(f - bg).max(-1) > thresh
        fg = ndimage.binary_opening(fg, structure=np.ones((3, 3), bool))
        lab, n = ndimage.label(fg)
        cur: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for c in range(1, n + 1):
            m = lab == c
            if m.sum() < min_area:
                continue
            if max_area is not None and m.sum() > max_area * (H / 128) ** 2:
                continue
            if max_elong is not None:
                ys, xs = np.nonzero(m)
                h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
                if max(h, w) > max_elong * min(h, w):
                    continue
            col = f[m].mean(0)
            k = contrast(f, m, fg)
            if k is not None and k < tol:
                continue                                       # ghost: looks like its surroundings
            best, best_iou = None, link_iou
            for gid, (pm, _c) in active.items():
                if gid in cur:
                    continue
                iou = (m & pm).sum() / (m | pm).sum()
                if iou >= best_iou:
                    best, best_iou = gid, iou
            if best is None:                                   # no overlap (lost, or moved past its own width):
                cm, bd = centroid(m), reid_dist * H            # the nearest track that looks the same
                cands = {**lost, **{gid: (centroid(pm), pc) for gid, (pm, pc) in active.items()}}
                for gid, (lc, lcol) in cands.items():
                    d = float(np.linalg.norm(cm - lc))
                    if gid not in cur and np.abs(col - lcol).max() < tol and d < bd:
                        best, bd = gid, d
            if best is None:
                if next_id > 255:
                    raise ValueError("track_seg: more than 255 groups in one clip")
                best, next_id = next_id, next_id + 1
            lost.pop(best, None)
            cur[best] = (m, col)
        taken = np.zeros((H, W), bool)
        for m, _c in cur.values():
            taken |= m
        for gid, (pm, pcol) in active.items():                 # unlinked tracks: at rest, or lost
            if gid in cur:
                continue
            m = pm & ~taken
            if m.sum() >= min_area and np.abs(f[m].mean(0) - pcol).max() < tol:
                k = contrast(f, m, fg | taken)
                if k is None or k >= tol:
                    cur[gid] = (m, pcol)
                    taken |= m
                    continue
            lost[gid] = (centroid(pm), pcol)
        grid = np.zeros((H, W), dtype=np.int32)
        for gid, (m, _c) in cur.items():
            grid[m] = gid
        blk = grid.reshape(G, patch, G, patch).transpose(0, 2, 1, 3).reshape(G, G, -1)
        for gid in cur:
            cnt = (blk == gid).sum(-1)
            sel = (cnt >= min_frac * blk.shape[-1]) & (cnt > ((blk > 0) & (blk != gid)).sum(-1))
            out[s][sel] = gid
        active = cur
    return out


class PossibleClips(torch.utils.data.IterableDataset):
    """Endless stream of possible clips (uint8 [T, res, res, 3]) from v2 shards, minus held-out seeds.
    Shards are split across workers and reshuffled every pass; a buffer shuffles clips within a worker."""

    def __init__(self, data: str, exclude_ids: set[str], res: int, split: str = "train", buffer: int = 256,
                 seed: int = 0, tokseg: bool = False, mix_data: str | None = None, mix_frac: float = 0.0):
        self.shards = sorted(glob.glob(os.path.join(data, split, "possible", "*.tar")))
        if not self.shards:
            raise FileNotFoundError(f"no possible shards under {data}/{split}")
        self.exclude, self.buffer, self.seed, self.res = exclude_ids, buffer, seed, res
        self.tokseg = tokseg  # also yield per-token object labels (shards from prep_cloud --tokseg)
        # mixing: each sample comes from mix_data with probability mix_frac. Every worker reads ALL mix shards (the
        # mix set is small), so batches mix at the sample level rather than shard by shard.
        if bool(mix_data) != (mix_frac > 0) or not 0 <= mix_frac < 1:
            raise ValueError("mix_data and 0 < mix_frac < 1 go together")
        self.mix_shards = sorted(glob.glob(os.path.join(mix_data, split, "possible", "*.tar"))) if mix_data else []
        if mix_data and not self.mix_shards:
            raise FileNotFoundError(f"no possible shards under {mix_data}/{split}")
        self.mix_frac = mix_frac

    def _stream(self, shards: list[str], rng: random.Random, wid: int):
        """Endless decoded clips from `shards`, reshuffled each pass, held-out ids skipped."""
        shards = list(shards)
        while True:
            rng.shuffle(shards)
            seen = 0
            # this class already splits shards across workers; WebDataset's own worker split would split again
            for s in wds.WebDataset(shards, shardshuffle=False, empty_check=False, workersplitter=lambda src: src):
                seen += 1
                if s["__key__"] in self.exclude:
                    continue
                frames = np.load(io.BytesIO(s["clip.npz"]))["frames"]
                clip = torch.from_numpy(downsample(frames, self.res))
                if self.tokseg:
                    if "tokseg.npy" not in s:
                        raise KeyError(f"{s['__key__']}: no tokseg.npy (prepare shards with prep_cloud --tokseg)")
                    clip = (clip, torch.from_numpy(np.load(io.BytesIO(s["tokseg.npy"]))))
                yield clip
            if seen == 0:
                raise RuntimeError(f"worker {wid}: a full pass over {shards} yielded no samples")

    def __iter__(self):
        torch.set_num_threads(1)
        wi = torch.utils.data.get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = random.Random(self.seed * 1000 + wid)
        mine = self.shards[wid::nw]
        if not mine:
            raise RuntimeError(f"worker {wid} has no shards ({len(self.shards)} shards, {nw} workers)")
        main = self._stream(mine, rng, wid)
        mix = self._stream(self.mix_shards, random.Random(self.seed * 1000 + wid + 500), wid) if self.mix_shards else None
        buf = []
        while True:
            src = mix if (mix is not None and rng.random() < self.mix_frac) else main
            buf.append(next(src))
            if len(buf) >= self.buffer:
                yield buf.pop(rng.randrange(len(buf)))


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
