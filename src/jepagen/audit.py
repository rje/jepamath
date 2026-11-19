"""Leak audit: can cheap, non-numerical image statistics predict the count?

Features come from the final frame (the reveal) of possible clips: total object pixels,
mean frame brightness, convex-hull area of object pixels, mean object size in pixels is
deliberately NOT included alone with total pixels (their ratio is the count). Timing feature:
reveal time. If any classifier beats chance by more than `margin`, the controls are leaking.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import webdataset as wds
from scipy.spatial import ConvexHull
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .shards import load_sample

FEATURES = ["total_object_px", "mean_brightness", "hull_area_px", "reveal_t"]


def _features(frames, seg, state) -> list[float]:
    f, s = frames[-1], seg[-1]
    obj = (s >= 1) & (s <= 250)
    ys, xs = np.nonzero(obj)
    hull = 0.0
    if len(xs) >= 3:
        try:
            hull = ConvexHull(np.stack([xs, ys], 1)).volume
        except Exception:  # degenerate (collinear) point sets have zero area
            hull = 0.0
    reveal = state["metrics"]["reveal_t"]
    return [float(obj.sum()), float(f.mean()), float(hull), float(reveal if reveal is not None else -1)]


def audit(data_dir: str, split: str, max_clips: int = 10_000, margin: float = 0.10) -> dict:
    paths = sorted(glob.glob(os.path.join(data_dir, split, "possible", "*.tar")))
    if not paths:
        raise FileNotFoundError(f"no shards under {data_dir}/{split}/possible")
    X, y = [], []
    for sample in wds.WebDataset(paths, shardshuffle=False):
        ep, frames, seg, state = load_sample(sample)
        X.append(_features(frames, seg, state))
        y.append(ep["meta"]["final"])
        if len(y) >= max_clips:
            break
    X, y = np.array(X), np.array(y)
    classes, counts = np.unique(y, return_counts=True)
    chance = counts.max() / counts.sum()
    folds = int(min(5, counts.min())) if counts.min() >= 2 else 2
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "gboost": HistGradientBoostingClassifier(max_iter=200),
    }
    res = {"n": int(len(y)), "classes": classes.tolist(), "chance": float(chance), "models": {}, "per_feature": {}}
    for name, mdl in models.items():
        acc = cross_val_score(mdl, X, y, cv=folds)
        res["models"][name] = {"acc_mean": float(acc.mean()), "acc_std": float(acc.std())}
    for j, fname in enumerate(FEATURES):
        acc = cross_val_score(HistGradientBoostingClassifier(max_iter=100), X[:, [j]], y, cv=folds)
        res["per_feature"][fname] = float(acc.mean())
    best = max(v["acc_mean"] for v in res["models"].values())
    res["leak"] = bool(best > chance + margin)
    res["margin"] = margin
    return res
