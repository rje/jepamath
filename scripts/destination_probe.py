"""Compare per-movie destination readouts and the existing native ending score."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

from jepatrain.placement_eval import intervals, scene
from jepatrain.reeval import load_run


MODES = ("prefix", "carry", "still", "landed")
PAIRS = {"shared_pickup": [0, 1], "shuffle_vs_add": [2, 1]}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(data, out, arm, records, stamp, device):
    weights = data / arm / "weights_100000.pt"
    enc, tgt, pred, step = load_run(str(weights.parent), weights.name, device=device)
    if step != 100000:
        raise ValueError("wrong endpoint")
    for m in (enc, tgt, pred):
        m.eval()
    all_features, all_distances = [], []
    for begin in range(0, len(records), 16):
        path = out / f"{arm}-{begin:03d}.npz"
        if not path.exists():
            features, distances = [], []
            for r in records[begin : begin + 16]:
                p = data / "scenes" / r["file"]
                if digest(p) != r["sha256"]:
                    raise ValueError(f"scene changed: {p}")
                collected = {}

                def collect(name, z):
                    if name in MODES:
                        # Fixed image quadrants of the last observed tubelet. No tray or route labels.
                        tiles = (
                            z[:, -(enc.grid**2) :]
                            .transpose(1, 2)
                            .reshape(len(z), -1, enc.grid, enc.grid)
                        )
                        collected[name] = (
                            F.adaptive_avg_pool2d(tiles, 2).flatten(1).cpu().numpy()
                        )

                with np.load(p) as z:
                    ds = scene(z, "", enc, tgt, pred, True, collect)
                features.append(np.stack([collected[m] for m in MODES]))
                distances.append(np.stack([ds[m] for m in (*MODES, "visible")]))
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                np.savez_compressed(
                    f, stamp=stamp, features=features, distances=distances
                )
            tmp.replace(path)
        with np.load(path) as z:
            if str(z["stamp"]) != stamp:
                raise ValueError("data or weights changed on resume")
            all_features.append(z["features"])
            all_distances.append(z["distances"])
        print(
            "scenes_done", begin + 16 + (arm == "treatment") * len(records), flush=True
        )
    return np.concatenate(all_features), np.concatenate(all_distances)


def hits(margins):
    signed = margins * np.array([-1, 1])
    return (signed > 1e-12).astype(float) + 0.5 * (abs(signed) <= 1e-12)


def run(data, out, device):
    manifest = data / "manifest.json"
    spec = json.loads(manifest.read_text())
    records = spec["rows"]
    seeds = [r["seed"] for r in records]
    if len(set(seeds)) != 192 or len(seeds) != 192:
        raise ValueError("unexpected number of independent scenes")
    fit = np.array([r["split"] == "fit" for r in records])
    starts = np.array([r["start"] for r in records])
    sides = np.array([r["hopper"] for r in records])
    for n in (1, 2):
        for side in ("left", "right"):
            subset = (starts == n) & (sides == side)
            if sum(subset & fit) != 16 or sum(subset & ~fit) != 32:
                raise ValueError("unbalanced design")
    out.mkdir(exist_ok=True, parents=True)
    results = {}
    raw = {}
    for arm in ("control", "treatment"):
        for name in ("args.json", "weights_100000.pt"):
            if digest(data / arm / name) != spec["models"][arm][name]:
                raise ValueError("checkpoint identity mismatch")
        features, distances = extract(data, out, arm, records, digest(manifest), device)
        raw[arm] = {}
        results[arm] = {}
        for pair, routes in PAIRS.items():
            detail = {}
            raw[arm][pair] = {}
            for i, mode in enumerate((*MODES, "visible")):
                native = (
                    distances[~fit, i][:, routes, 0] - distances[~fit, i][:, routes, 1]
                )
                modes = {"native": native}
                if mode != "visible":
                    x = features[:, i][:, routes]
                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=0.01, solver="liblinear", random_state=0),
                    )
                    model.fit(
                        x[fit].reshape(-1, x.shape[-1]), np.tile([0, 1], fit.sum())
                    )
                    modes["probe"] = model.decision_function(
                        x[~fit].reshape(-1, x.shape[-1])
                    ).reshape(-1, 2)
                for method, margin in modes.items():
                    key = method + "_" + mode
                    raw[arm][pair][key] = margin.tolist()
                    h = hits(margin)
                    item = dict(accuracy=intervals(h.mean(1)), per_route=intervals(h))
                    if mode == "carry":
                        item["by_start"] = {
                            str(n): intervals(h[starts[~fit] == n].mean(1))
                            for n in (1, 2)
                        }
                        item["by_hopper_side"] = {
                            s: intervals(h[sides[~fit] == s].mean(1))
                            for s in ("left", "right")
                        }
                    detail[key] = item
            for a, b in [
                ("probe_carry", "native_carry"),
                ("probe_carry", "probe_still"),
                ("native_carry", "native_still"),
            ]:
                gains = hits(np.array(raw[arm][pair][a])) - hits(
                    np.array(raw[arm][pair][b])
                )
                detail[a + "_minus_" + b] = intervals(gains.mean(1))
            results[arm][pair] = detail
    (out / "movie-margins.json").write_text(
        json.dumps(dict(test_seeds=np.array(seeds)[~fit].tolist(), arms=raw))
    )
    (out / "diagnostic.json").write_text(
        json.dumps(
            dict(
                fit_scenes=64, test_scenes=128, manifest=digest(manifest), arms=results
            ),
            indent=1,
        )
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--auto-resume", action="store_true", help="reuse verified feature blocks in an existing output directory")
    a = ap.parse_args()
    if not a.auto_resume and a.out.exists() and any(a.out.iterdir()):
        ap.error("use a fresh output directory or --auto-resume")
    run(a.data, a.out, a.device)
