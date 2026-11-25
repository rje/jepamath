"""Training shards for cloud runs: possible clips at --res px (downsampled if rendered larger), frames + episode only.

    python -m jepatrain.prep_cloud --src /mnt/data/jepamath/v3 --out /mnt/data/jepamath/v3_256 --res 256

Same layout as v2 (<out>/train/possible/shard-*.tar) and the same `downsample` the trainer would apply on
the fly, so PossibleClips/probe read either one identically. Drops seg masks and state.json (unused by
training) and so cuts the upload and every instance's download several-fold. CPU only.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
from multiprocessing import Pool

import numpy as np


def convert(args: tuple[str, str, int, int | None, int, str]) -> tuple[str, int]:
    import torch
    import webdataset as wds

    from .data import downsample, motion_seg, token_seg, track_seg
    torch.set_num_threads(1)
    src, dst, res, patch, tubelet, labels = args
    tmp = dst + ".tmp"
    n = 0
    with wds.TarWriter(tmp) as w:
        for s in wds.WebDataset([src], shardshuffle=False, empty_check=False):
            z = np.load(io.BytesIO(s["clip.npz"]))
            frames = downsample(z["frames"], res)
            buf = io.BytesIO()
            np.savez_compressed(buf, frames=frames)
            out = {"__key__": s["__key__"], "clip.npz": buf.getvalue(), "episode.json": s["episode.json"]}
            if patch is not None:
                tb = io.BytesIO()
                if labels == "render":
                    lab = token_seg(z["seg"], res, patch, tubelet)
                elif labels == "motion":
                    lab = motion_seg(frames, patch, tubelet)    # pixels only, at the training resolution
                elif labels == "track":
                    lab = track_seg(frames, patch, tubelet)     # pixels only: motion + persistence + hand-sized prior
                elif labels == "track-noprior":
                    lab = track_seg(frames, patch, tubelet, max_area=None, max_elong=None)  # S2 attribution arm
                else:
                    raise ValueError(f"unknown labels {labels!r}")
                np.save(tb, lab)
                out["tokseg.npy"] = tb.getvalue()
            w.write(out)
            n += 1
    os.replace(tmp, dst)
    return os.path.basename(dst), n


def main() -> int:
    ap = argparse.ArgumentParser(prog="jepatrain.prep_cloud")
    ap.add_argument("--src", default="/mnt/data/jepamath/v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--procs", type=int, default=22)
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--tokseg", type=int, nargs=2, metavar=("PATCH", "TUBELET"), default=None,
                    help="also store per-token object labels for the object-label branch (encoder patch and tubelet)")
    ap.add_argument("--labels", choices=("render", "motion", "track", "track-noprior"), default="render",
                    help="render: the renderer's object ids (token_seg); motion: label-free groups from motion "
                         "(motion_seg, pixels only); track: the S2 tracking grouper (track_seg: motion + persistence "
                         "+ hand-sized prior, pixels only); track-noprior: track_seg without the hand-sized prior. "
                         "All stored in the same tokseg.npy slot")
    a = ap.parse_args()
    if a.labels != "render" and not a.tokseg:
        raise SystemExit("--labels needs --tokseg PATCH TUBELET")
    patch, tubelet = a.tokseg if a.tokseg else (None, 0)
    shards = sorted(glob.glob(os.path.join(a.src, a.split, "possible", "*.tar")))
    if not shards:
        raise FileNotFoundError(f"no shards under {a.src}/{a.split}/possible")
    od = os.path.join(a.out, a.split, "possible")
    os.makedirs(od, exist_ok=True)
    jobs = [(s, os.path.join(od, os.path.basename(s)), a.res, patch, tubelet, a.labels) for s in shards if not os.path.exists(os.path.join(od, os.path.basename(s)))]
    total = 0
    with Pool(min(a.procs, max(1, len(jobs)))) as p:
        for name, n in p.imap_unordered(convert, jobs):
            total += n
            print(f"{name}: {n} clips", flush=True)
    counts = {}
    import webdataset as wds
    for s in sorted(glob.glob(os.path.join(od, "*.tar"))):
        counts[os.path.basename(s)] = sum(1 for _ in wds.WebDataset([s], shardshuffle=False, empty_check=False))
    with open(os.path.join(a.out, f"manifest-{a.split}-possible.json"), "w") as f:
        json.dump({"source": a.src, "res": a.res, "tokseg": a.tokseg, "labels": a.labels if a.tokseg else None, "shards": counts, "clips": sum(counts.values())}, f, indent=1)
    print(f"{sum(counts.values())} clips in {len(counts)} shards -> {od}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
