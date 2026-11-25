"""Pictures of object sense in frozen encoders (exploratory, for the object-sense report; not a battery metric).

On v4b_probe clips (test part, index >= 140, as in the battery's probe split), for each checkpoint, at layer 8:
  sim/      : cosine similarity of every token in a slot to one anchor token on an object, shown as a rank map
              (the share of the slot's tokens it beats), with each object's true token outline
  ident/    : the same anchor's similarity to the tokens of a later slot (slot + 4), after the object moved
  pca/      : the clip's layer-8 tokens projected on their top 3 principal components, as RGB
  fig/      : a logistic figure-vs-background probe (fit on clips 0-99) applied to the clip
plus scene panels (frame, renderer labels, track_seg labels). Also writes stats.json with, per panel, precision@k: of
the k tokens most similar to the anchor (k = the anchor object's token count), the share on the anchor's object.

    CUDA_VISIBLE_DEVICES= python scripts/viz_object_sense.py --out DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import webdataset as wds
from matplotlib import colormaps
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from jepagen.shards import load_sample  # noqa: E402
from jepatrain.data import downsample, token_seg, track_seg  # noqa: E402
from jepatrain.reeval import load_run  # noqa: E402
from locate_surprise import MEAN, STD  # noqa: E402
from object_sense import layer_tokens  # noqa: E402

B = "/mnt/data/jepamath/trackb"
MODELS = {"untrained": (f"{B}/cloud-v4b-128-cw50-only", "weights_000000.pt"),
          "plain": (f"{B}/cloud-v4b-128-cw50-only", "weights_030000.pt"),
          "track": (f"{B}/cloud-v4b-128-track", "weights_030000.pt"),
          "objlabel": (f"{B}/cloud-v4b-128-mask05b", "weights_030000.pt")}
PROBE = "/mnt/data/jepamath/v4b_probe"
LAYER, RES, P, TB, G, S, GAP = 8, 128, 8, 4, 16, 8, 4
UP = 2                                                # display scale: 128 px -> 256 px
PALETTE = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (0, 114, 178), (213, 94, 0),
           (204, 121, 167), (255, 255, 255)]


def clips(lo, hi):
    for i, s in enumerate(wds.WebDataset(sorted(glob.glob(os.path.join(PROBE, "train", "possible", "*.tar"))),
                                         shardshuffle=False, empty_check=False)):
        if i >= hi:
            return
        if i >= lo:
            ep, frames, seg, state = load_sample(s)
            fr = downsample(frames, RES)
            yield i, fr, token_seg(seg, RES, P, TB).astype(int)


def frame_img(fr, sl):
    return Image.fromarray(fr[sl * TB + TB // 2]).resize((RES * UP, RES * UP), Image.NEAREST)


def outline(img, mask, color, width=2):
    """Draw the boundary of a token mask [G, G] on img."""
    d, c = ImageDraw.Draw(img), RES * UP // G
    for y in range(G):
        for x in range(G):
            if not mask[y, x]:
                continue
            if y == 0 or not mask[y - 1, x]: d.line([(x * c, y * c), ((x + 1) * c, y * c)], fill=color, width=width)
            if y == G - 1 or not mask[y + 1, x]: d.line([(x * c, (y + 1) * c), ((x + 1) * c, (y + 1) * c)], fill=color, width=width)
            if x == 0 or not mask[y, x - 1]: d.line([(x * c, y * c), (x * c, (y + 1) * c)], fill=color, width=width)
            if x == G - 1 or not mask[y, x + 1]: d.line([((x + 1) * c, y * c), ((x + 1) * c, (y + 1) * c)], fill=color, width=width)
    return img


def heat(base, vals, cmap="magma", alpha=0.72):
    """Blend a [G, G] map in [0, 1] over a frame."""
    rgb = (colormaps[cmap](vals)[..., :3] * 255).astype(np.uint8)
    h = Image.fromarray(rgb).resize(base.size, Image.NEAREST)
    return Image.blend(base.convert("RGB"), h, alpha)


def labels_img(base, lab):
    """Colour each labelled token by its label id (0 left clear)."""
    a = np.array(base.convert("RGB")).astype(float)
    c = RES * UP // G
    for y in range(G):
        for x in range(G):
            if lab[y, x]:
                col = np.array(PALETTE[(lab[y, x] - 1) % len(PALETTE)])
                a[y * c:(y + 1) * c, x * c:(x + 1) * c] = 0.35 * a[y * c:(y + 1) * c, x * c:(x + 1) * c] + 0.65 * col
    return Image.fromarray(a.astype(np.uint8))


def rank(v):
    return np.argsort(np.argsort(v)) / (len(v) - 1)


def pick_anchor(lab, sl):
    """The token nearest the centroid of the largest object in slot sl, and that object's label."""
    ids, cnt = np.unique(lab[sl][lab[sl] > 0], return_counts=True)
    o = ids[cnt.argmax()]
    ys, xs = np.nonzero(lab[sl] == o)
    k = np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2)
    return o, ys[k] * G + xs[k]


def choose_clips(n):
    """Test clips with >= 3 objects of >= 3 tokens at slot 2, and one of them moving >= 2 tokens by slot 6."""
    out = []
    for i, fr, lab in clips(140, 400):
        sl = 2
        ids, cnt = np.unique(lab[sl][lab[sl] > 0], return_counts=True)
        if (cnt >= 3).sum() < 3:
            continue
        moved = None
        for o in ids[cnt >= 3]:
            a, b = (np.argwhere(lab[t] == o) for t in (sl, sl + GAP))
            if len(b) >= 3 and np.abs(a.mean(0) - b.mean(0)).max() >= 2:
                moved = int(o); break
        if moved is None:
            continue
        out.append((i, fr, lab, moved))
        if len(out) == n:
            return out
    raise ValueError(f"only {len(out)} clips qualify")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("run with CUDA_VISIBLE_DEVICES=")
    torch.set_num_threads(16)
    for d in ("scene", "sim", "ident", "pca", "fig"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    chosen = choose_clips(a.n)
    stats = {"clips": [c[0] for c in chosen], "panels": {}}
    sl = 2
    for ci, (idx, fr, lab, moved) in enumerate(chosen):
        L = lab.reshape(S, G, G)
        f0, f1 = frame_img(fr, sl), frame_img(fr, sl + GAP)
        o, anc = pick_anchor(L, sl)
        sc = f0.copy(); outline(sc, L[sl] == o, (255, 255, 255), 3)
        d = ImageDraw.Draw(sc); c = RES * UP // G; ay, ax = divmod(anc, G)
        d.ellipse([ax * c + 3, ay * c + 3, (ax + 1) * c - 3, (ay + 1) * c - 3], fill=(255, 255, 255))
        sc.save(f"{a.out}/scene/c{ci}_frame.png")
        f1.save(f"{a.out}/scene/c{ci}_frame_later.png")
        labels_img(f0, L[sl]).save(f"{a.out}/scene/c{ci}_render.png")
        tr = track_seg(fr, P, TB).astype(int)
        labels_img(f0, tr[sl]).save(f"{a.out}/scene/c{ci}_track.png")
        # identity anchor: the moved object at slot sl
        lm = lab.reshape(S, -1)
        ys, xs = np.nonzero(L[sl] == moved)
        k = np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2); anc_m = ys[k] * G + xs[k]
        im = f0.copy(); outline(im, L[sl] == moved, (255, 255, 255), 3)
        my, mx = divmod(anc_m, G)
        ImageDraw.Draw(im).ellipse([mx * c + 3, my * c + 3, (mx + 1) * c - 3, (my + 1) * c - 3], fill=(255, 255, 255))
        im.save(f"{a.out}/scene/c{ci}_moved.png")
        stats["panels"][ci] = {"clip": idx, "anchor_obj_tokens": int((L[sl] == o).sum()),
                               "moved_obj_tokens_later": int((L[sl + GAP] == moved).sum()),
                               "objects": int(len(np.unique(L[sl][L[sl] > 0]))),
                               "track_recall_slot": float(((tr[sl] > 0) & (L[sl] > 0)).sum() / max((L[sl] > 0).sum(), 1))}
    for name, (run, w) in MODELS.items():
        enc, _t, _p, step = load_run(run, w, device="cpu"); enc.eval()
        # figure probe: fit on clips 0-99, 40 tokens per slot
        Xf, yf = [], []
        rng = np.random.default_rng(0)
        for i, fr, lab in clips(0, 100):
            x = (torch.from_numpy(fr)[None].permute(0, 4, 1, 2, 3).float() / 255 - MEAN) / STD
            t = layer_tokens(enc, x, {LAYER})[LAYER].float().numpy()
            k = rng.choice(len(t), 320, replace=False)
            Xf.append(t[k]); yf.append(lab.reshape(-1)[k] > 0)
        probe = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced").fit(np.concatenate(Xf), np.concatenate(yf))
        for ci, (idx, fr, lab, moved) in enumerate(chosen):
            L = lab.reshape(S, G, G); lm = lab.reshape(S, -1)
            x = (torch.from_numpy(fr)[None].permute(0, 4, 1, 2, 3).float() / 255 - MEAN) / STD
            raw = layer_tokens(enc, x, {LAYER})[LAYER].float()
            t = torch.nn.functional.normalize(raw, dim=-1).numpy().reshape(S, G * G, -1)
            f0, f1 = frame_img(fr, sl), frame_img(fr, sl + GAP)
            o, anc = pick_anchor(L, sl)
            sim = t[sl] @ t[sl][anc]
            im = heat(f0, rank(sim).reshape(G, G))
            for oid in np.unique(L[sl][L[sl] > 0]):
                outline(im, L[sl] == oid, (255, 255, 255) if oid == o else (120, 200, 255), 3 if oid == o else 1)
            im.save(f"{a.out}/sim/c{ci}_{name}.png")
            kk = int((lm[sl] == o).sum()); top = np.argsort(-sim)[:kk]
            pr = stats["panels"][ci]
            pr[f"{name}_sim_p@k"] = float((lm[sl][top] == o).mean())
            pr[f"{name}_sim_obj_mean"] = float(sim[lm[sl] == o].mean()); pr[f"{name}_sim_rest_mean"] = float(sim[lm[sl] != o].mean())
            # identity across time
            ys, xs = np.nonzero(L[sl] == moved)
            k = np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2); anc_m = ys[k] * G + xs[k]
            s2 = t[sl + GAP] @ t[sl][anc_m]
            im = heat(f1, rank(s2).reshape(G, G))
            for oid in np.unique(L[sl + GAP][L[sl + GAP] > 0]):
                outline(im, L[sl + GAP] == oid, (255, 255, 255) if oid == moved else (120, 200, 255), 3 if oid == moved else 1)
            im.save(f"{a.out}/ident/c{ci}_{name}.png")
            kk = int((lm[sl + GAP] == moved).sum()); top = np.argsort(-s2)[:kk]
            pr[f"{name}_ident_p@k"] = float((lm[sl + GAP][top] == moved).mean())
            # PCA of the clip's tokens
            X = raw.numpy() - raw.numpy().mean(0)
            _u, _s, vt = np.linalg.svd(X, full_matrices=False)
            pc = (X @ vt[:3].T).reshape(S, G * G, 3)[sl]
            lo, hi = np.percentile(pc, 2, 0), np.percentile(pc, 98, 0)
            rgb = (np.clip((pc - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8).reshape(G, G, 3)
            pim = Image.fromarray(rgb).resize(f0.size, Image.NEAREST)
            for oid in np.unique(L[sl][L[sl] > 0]):
                outline(pim, L[sl] == oid, (255, 255, 255), 1)
            pim.save(f"{a.out}/pca/c{ci}_{name}.png")
            # figure probe
            pf = probe.predict_proba(raw.numpy().reshape(S, G * G, -1)[sl])[:, 1]
            im = heat(f0, pf.reshape(G, G), cmap="viridis")
            for oid in np.unique(L[sl][L[sl] > 0]):
                outline(im, L[sl] == oid, (255, 255, 255), 1)
            im.save(f"{a.out}/fig/c{ci}_{name}.png")
            pr[f"{name}_fig_bacc"] = float(0.5 * ((pf[lm[sl] > 0] > 0.5).mean() + (pf[lm[sl] == 0] <= 0.5).mean()))
        print(f"{name} done (step {step})", flush=True)
    json.dump(stats, open(f"{a.out}/stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
