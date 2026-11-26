"""Report figures from real C9 test clips (objlabel-100k): what the model sees, what its prediction says, where it goes wrong.

Fits P1's ridge count probe on C8 predicted features (cached), picks a C9 v1 block where the native ending choice gets
the conflict edge wrong while the probe orders the counts correctly, and draws:
  fig_sees.png     the H history as the model receives it (hopper masked, screen up), then the revealed ending
  fig_predicts.png true ending, the two candidate endings with the model's mismatch to each (choice boxed),
                   the per-patch mismatch between the prediction and the true ending, and the probe's count
  fig_wrong.png    H (3 objects, less material) vs L-remove (2 objects, more material): both pick the same ending;
                   the count read from the prediction separates them

    CUDA_VISIBLE_DEVICES= python scripts/viz_report.py --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from c6_score import load_model, mask_hopper  # noqa: E402
from p1_count_probe import c8_items, feats, ridge_cv  # noqa: E402
from p2_controls import model_feats  # noqa: E402

B = "/mnt/data/jepamath/trackb"
RUN, W = f"{B}/cloud-v4b-128-mask05-100k", "weights_100000.pt"
C8, C9 = "/mnt/data/jepamath/c8_v1/manifest.json", "/mnt/data/jepamath/c9_v1/manifest.json"
MATS = f"{B}/c9/c9_v1_objlabel-100k_mats.npz"
BLUE, RED, GREY = "#2f6fd6", "#c8423b", "#8a8a8a"


def fit_probe(enc, tgt, pred, cache):
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    fm = json.load(open(C8))
    X, y, g = [], [], []
    for seed, s, z, fin in c8_items(fm):
        m = mask_hopper(z["frames"], z["hopper_px"], z["hopper_ring"], s["ctx"])
        P, *_ = feats(z["frames"], m, s["ctx"], s["first_post_slot"], z["box_tokens"].astype(bool), enc, tgt, pred)
        X.append(P); y.append(fin); g += [seed] * len(fin)
    probe, info = ridge_cv(np.concatenate(X), np.concatenate(y), np.array(g))
    pickle.dump((probe, info), open(cache, "wb"))
    return probe, info


@torch.no_grad()
def mismatch_map(frames, masked, ctx, first, box, enc, tgt, pred):
    """Per-patch mean L1 between predicted and actual (target-encoded) post-reveal tokens, [16, 16] (NaN off the box)."""
    from p2_controls import prep
    import torch.nn.functional as F
    G = enc.grid * enc.grid
    xm, xf = prep(masked[None]), prep(frames[None])
    slots = xm.shape[2] // 4
    n_ctx, N = ctx // 4 * G, slots * G
    tok = torch.from_numpy(np.flatnonzero(box.ravel()))
    keep = torch.cat([s * G + tok for s in range(ctx // 4 + first, slots)])
    P = pred(enc(xm[:, :, :ctx]), torch.arange(n_ctx)[None], torch.arange(n_ctx, N)[None]).float()[:, keep - n_ctx]
    T = F.layer_norm(tgt(xf, keep[None]).float(), (P.size(-1),))
    d = (P - T).abs().mean(-1)[0].reshape(-1, len(tok)).mean(0).numpy()
    out = np.full(G, np.nan)
    out[tok.numpy()] = d
    return out.reshape(enc.grid, enc.grid)


def big(img, k=3):
    return np.kron(img, np.ones((k, k, 1), dtype=img.dtype))


def ax_img(ax, img, title=None, color="#222", edge=None, lw=3):
    ax.imshow(big(img))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(edge is not None)
        if edge:
            s.set_edgecolor(edge); s.set_linewidth(lw)
    if title:
        ax.set_title(title, fontsize=10, color=color, pad=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if torch.cuda.is_available():
        raise SystemExit("run with CUDA_VISIBLE_DEVICES=")
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(12)
    enc, tgt, pred, _ = load_model(RUN, W)
    for m in (enc, tgt, pred):
        m.eval()
    probe, info = fit_probe(enc, tgt, pred, os.path.join(a.out, "probe_objlabel100k.pkl"))
    print("probe", info, flush=True)
    man, M = json.load(open(C9)), np.load(MATS)
    srec = {json.loads(l)["seed"]: json.loads(l) for l in open(os.path.join(man["sources_dir"], "items.jsonl"))}
    pick = None
    for i, b in enumerate(man["blocks"]):
        s = srec[b["source"]]
        assert M["source"][i] == b["source"]
        if b["shape"] != "sphere":
            continue
        n = s["names"]
        h, l = n.index("orig:H"), n.index("orig:L-remove")
        R = M["fwd"][i]                                   # [12 clips, 2 endings (2-obj, 3-obj)] L1 mismatch
        # 'fwd' donor: 2-object ending = 2 small compact, 3-object ending = 3 big wide, so a material reader gets both
        # wrong. Selected: both picks wrong (H -> 2 small, L-remove -> 3 big), probe orders the counts correctly
        if not (R[h, 0] < R[h, 1] and R[l, 1] < R[l, 0]):
            continue
        z = np.load(os.path.join(man["sources_dir"], f"source_{b['source']:09d}.npz"))
        masked = mask_hopper(z["frames"], z["hopper_px"], z["hopper_ring"], s["ctx"])
        _, _, pp, _ = model_feats(masked, s["ctx"], s["first_post_slot"], z["box_tokens"].astype(bool), enc, pred)
        c = probe.predict(pp)
        if c[h] - c[l] > 0.6:
            pick = (i, b, s, z, masked, c, h, l, R)
            break
    if pick is None:
        raise SystemExit("no block matched the selection rule")
    i, b, s, z, masked, c, h, l, R = pick
    print(f"block {i} source {b['source']} counts H {c[h]:.2f} L-remove {c[l]:.2f}", flush=True)
    ctx, first = s["ctx"], s["first_post_slot"]
    zd = np.load(os.path.join(man["fwd_dir"], f"donor_{b['fwd']['donor']:09d}.npz"))
    endings = {2: zd["frames"][0][-1], 3: zd["frames"][1][-1]}
    tr = s["transfers"]["orig:H"]
    fps = 4

    def carry(name, k):
        v = s["transfers"][name][k]["visible_frames"]
        if not v:
            raise ValueError(f"{name} transfer {k}: carried object never visible")
        return v[len(v) // 2]

    def move_label(name, k):
        src, dst = s["meta"][name]["route"][k]
        return f"move {k + 1}: {src} to {dst}"

    # ---- fig 1: what the model sees ----
    idx = [0, carry("orig:H", 0), int(tr[0]["t_release"] * fps), carry("orig:H", 1), int(tr[1]["t_release"] * fps), ctx - 1]
    labels = ["start", move_label("orig:H", 0), "move 1 done", move_label("orig:H", 1), "move 2 done", "last frame it sees"]
    fig, axs = plt.subplots(1, 7, figsize=(15, 2.9))
    for ax, k, lab in zip(axs, idx, labels):
        ax_img(ax, masked[h][k], lab)
    ax_img(axs[-1], z["frames"][h][-1], "hidden: the real ending", color=BLUE, edge=BLUE)
    fig.suptitle("What the model sees: hopper blanked out, the box behind a screen. It must work out the ending.",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(a.out, "fig_sees.png"), dpi=110, facecolor="white")
    plt.close(fig)

    # ---- fig 2: what its prediction says ----
    mm = mismatch_map(z["frames"][h], masked[h], ctx, first, z["box_tokens"].astype(bool), enc, tgt, pred)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    ax_img(axs[0], z["frames"][h][-1], "real ending: 3 small objects")
    ch = 0 if R[h, 0] < R[h, 1] else 1
    for ax, e, k in ((axs[1], 2, 0), (axs[2], 3, 1)):
        good = (e == 3)
        ax_img(ax, endings[e], f"candidate: {e} {'big' if e == 3 else 'small'} objects\nmismatch {R[h, k]:.4f}" + ("   <- picked" if k == ch else ""),
               color=(RED if k == ch and not good else "#222"), edge=(RED if k == ch else None))
    axs[3].imshow(big(z["frames"][h][-1]))
    hm = np.kron(mm, np.ones((24, 24)))
    im = axs[3].imshow(np.ma.masked_invalid(hm), cmap="magma", alpha=0.6)
    axs[3].set_xticks([]); axs[3].set_yticks([])
    axs[3].set_title(f"where its expectation differs\ncount readout {c[h]:.2f} (true 3)", fontsize=10,
                     color=BLUE)
    fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.02).ax.tick_params(labelsize=8)
    fig.suptitle(f"What it predicts: its prediction ranks this history above the 2-object one (count readout {c[h]:.2f} "
                 f"vs {c[l]:.2f}), yet it picks the ending with 2 small objects", fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.86, "Candidate endings come from other scenes whose box overlaps this one by at least 60%; only that region is "
             "compared. Heat map: brighter = its prediction differs more from the real ending.", fontsize=9.5, color="#555")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(os.path.join(a.out, "fig_predicts.png"), dpi=110, facecolor="white")
    plt.close(fig)

    # ---- fig 3: what it gets wrong ----
    fig, axs = plt.subplots(2, 5, figsize=(15.5, 7.4), gridspec_kw={"hspace": 0.55})
    rows = ((h, 3, "3 small objects"), (l, 2, "2 objects, one big"))
    for r, (j, true_n, desc) in enumerate(rows):
        nm = s["names"][j]
        for k in (0, 1):
            ax_img(axs[r, k], masked[j][carry(nm, k)], move_label(nm, k))
        ax_img(axs[r, 2], z["frames"][j][-1], f"real ending: {desc}")
        chj = 0 if R[j, 0] < R[j, 1] else 1
        ok = (chj == 1) == (true_n == 3)
        ax_img(axs[r, 3], endings[2 if chj == 0 else 3],
               f"its pick: {'2 small' if chj == 0 else '3 big'} objects\n" + ("right" if ok else "wrong"),
               color=(BLUE if ok else RED), edge=(BLUE if ok else RED))
        axs[r, 4].axis("off")
        axs[r, 4].text(0.05, 0.60, f"{c[j]:.2f}", fontsize=34, color=BLUE, weight="bold", transform=axs[r, 4].transAxes)
        axs[r, 4].text(0.05, 0.36, f"count readout\n(true count {true_n})", fontsize=10, color="#444",
                       transform=axs[r, 4].transAxes)
    fig.suptitle("What it gets wrong: both picks follow material (less material -> fewer objects), "
                 "while the count readout still ranks the histories correctly", fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(a.out, "fig_wrong.png"), dpi=110, facecolor="white")
    plt.close(fig)
    json.dump({"block": i, "source": b["source"], "counts": {"H": float(c[h]), "L-remove": float(c[l])},
               "mismatch_rev": R.tolist(), "names": s["names"]}, open(os.path.join(a.out, "picked.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
