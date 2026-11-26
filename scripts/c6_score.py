"""C6 scoring (docs/2026-11-26-c6-joint-count-design-draft.md): one 7 x 3 matrix per frozen source-donor block.

Rows: the source's seven histories (c6_items.CELLS order; finals 1 2 1 2 3 1 2). Columns: the donor's endings
with 1, 2, 3 objects. Model score S[h, e] = mean tokenwise L1 between the predictor's output for history h
(hopper masked in every context frame, queried with all future tokens, post-reveal tokens of the block's
common box mask selected) and the target encoder's layer-normed tokens of donor ending e on the same tokens.

Endpoints (each in [0, 1]):
  I    rows N1- N1A N2- N2A   (vary initial count and add slot; remove slot BB)
  II   rows N1A N1AR N2A N2AR (vary initial count and remove slot; add slot HB)
  III  rows N2- N2A N2R N2AR  (vary both slots; N = 2)
       credit per row: P(chosen ending is the 2-object one), uniform over tied minimum columns; that if the true
       final is 2, else 1 minus it. Equal row weights.
  1v3  rows whose final is 1 (N1- N1AR N2R) or 3 (N2A): is the true extreme closer than the other extreme
       (tie 0.5); final-1 rows and the final-3 row weighted equally.
  exact  strict unique argmin == true final, over all 7 rows (reported; chance 1/3).
  mean_I_II, mean_I_III: equal-weight means.
Bootstrap: 10,000 resamples of whole blocks, stratified by add_first, all endpoints and models in each draw.
Readers: symbolic ceilings (best decoder from one feature set, per comparison), the state counter (must be 1),
and visual / metadata shortcut matrices scored like a model.

    python scripts/c6_score.py --manifest M.json --out OUT.json --run R --weights W --name N [...]
    (--weights untrained: the run's architecture at its random init, seed 0)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from c6_items import CELLS  # noqa: E402
from locate_surprise import MEAN, STD  # noqa: E402

TUB = 4
FIN = np.array([N + a - r for _, N, a, r in CELLS])
COMP = {"I": [0, 1, 3, 4], "II": [1, 2, 4, 6], "III": [3, 4, 5, 6]}
ONE, THREE = [i for i in range(7) if FIN[i] == 1], [i for i in range(7) if FIN[i] == 3]
assert [CELLS[i][0] for i in COMP["I"]] == ["N1-", "N1A", "N2-", "N2A"]
assert [CELLS[i][0] for i in COMP["II"]] == ["N1A", "N1AR", "N2A", "N2AR"]
assert [CELLS[i][0] for i in COMP["III"]] == ["N2-", "N2A", "N2R", "N2AR"]


# ---------------- endpoints on one block ----------------

TIE_ATOL = 1e-12


def tied(row: np.ndarray) -> np.ndarray:
    """THE tie predicate for every endpoint: columns within TIE_ATOL of the row minimum."""
    return np.flatnonzero(np.abs(row - row.min()) <= TIE_ATOL)


def p_two(row: np.ndarray) -> float:
    m = tied(row)
    return float(1 in m) / len(m)


def block_endpoints(S: np.ndarray) -> dict[str, float]:
    out = {}
    for k, rows in COMP.items():
        out[k] = float(np.mean([p_two(S[i]) if FIN[i] == 2 else 1 - p_two(S[i]) for i in rows]))

    def ext(i):
        t, o = (0, 2) if FIN[i] == 1 else (2, 0)
        return 0.5 if abs(S[i, t] - S[i, o]) <= TIE_ATOL else float(S[i, t] < S[i, o])
    out["1v3"] = 0.5 * np.mean([ext(i) for i in ONE]) + 0.5 * np.mean([ext(i) for i in THREE])
    out["exact"] = float(np.mean([len(tied(S[i])) == 1 and S[i].argmin() == FIN[i] - 1 for i in range(7)]))
    out["tie_rate"] = float(np.mean([len(tied(S[i])) > 1 for i in range(7)]))
    out["mean_I_II"] = (out["I"] + out["II"]) / 2
    out["mean_I_III"] = (out["I"] + out["II"] + out["III"]) / 3
    return out


def confusion(mats: list[np.ndarray]) -> list[list[float]]:
    """[true final, chosen ending], tied minima sharing one row's credit equally."""
    c = np.zeros((3, 3))
    for S in mats:
        for i in range(7):
            m = tied(S[i])
            c[FIN[i] - 1, m] += 1 / len(m)
    return c.round(2).tolist()


def bootstrap(per_block: dict[str, list[dict]], strata: np.ndarray, n: int = 10_000, seed: int = 0,
              pairs: list[tuple[str, str]] = ()) -> dict:
    """per_block[name] = list of block endpoint dicts (same block order for every name)."""
    rng = np.random.default_rng(seed)
    keys = list(next(iter(per_block.values()))[0])
    arr = {nm: np.array([[b[k] for k in keys] for b in bl]) for nm, bl in per_block.items()}
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    draws = {nm: [] for nm in arr}
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        for nm, A in arr.items():
            draws[nm].append(A[idx].mean(0))
    out = {}
    for nm, A in arr.items():
        D = np.array(draws[nm])
        out[nm] = {k: {"est": float(A[:, j].mean()), "ci": [float(np.percentile(D[:, j], 2.5)),
                                                             float(np.percentile(D[:, j], 97.5))]}
                   for j, k in enumerate(keys)}
    out["paired"] = {}
    for a, b in pairs:
        D = np.array(draws[a]) - np.array(draws[b])
        out["paired"][f"{a} - {b}"] = {k: {"est": float((arr[a] - arr[b])[:, j].mean()),
                                           "ci": [float(np.percentile(D[:, j], 2.5)), float(np.percentile(D[:, j], 97.5))]}
                                       for j, k in enumerate(keys)}
    return out


# ---------------- symbolic readers: best-decoder ceilings ----------------

def features(add_first: bool) -> dict[str, list]:
    """Per row, the information each count-free reader has. Slot routes in temporal order."""
    rows = []
    for _, N, a, r in CELLS:
        add_r, rem_r = ("HB" if a else "BB"), ("BH" if r else "BB")
        s1, s2 = (add_r, rem_r) if add_first else (rem_r, add_r)
        rows.append({"N": N, "s1": s1, "s2": s2, "a": a, "r": r})
    f = {"ending_only": [0] * 7, "N": [x["N"] for x in rows], "first_transfer": [x["s1"] for x in rows],
         "last_transfer": [x["s2"] for x in rows], "route": [(x["s1"], x["s2"]) for x in rows],
         "net": [x["a"] - x["r"] for x in rows], "sign": [int(np.sign(x["a"] - x["r"])) for x in rows],
         "n_in_out": [(x["a"], x["r"]) for x in rows], "n_shuffles": [2 - x["a"] - x["r"] for x in rows],
         "N+add_slot": [(x["N"], x["a"]) for x in rows], "N+remove_slot": [(x["N"], x["r"]) for x in rows],
         "parity": [(x["N"] + x["a"] + x["r"]) % 2 for x in rows], "state": list(FIN)}
    return f


def ceiling(feat: list, rows: list[int]) -> float:
    """Best accuracy on 'final is 2' for a reader that sees only `feat` (equal row weights)."""
    tot = 0.0
    for v in {feat[i] for i in rows}:
        g = [i for i in rows if feat[i] == v]
        p = np.mean([FIN[i] == 2 for i in g])
        tot += len(g) * max(p, 1 - p)
    return tot / len(rows)


def ceiling_1v3(feat: list) -> float:
    def acc(rows_true, rows_other):
        return rows_true
    groups = {}
    for i in ONE + THREE:
        groups.setdefault(feat[i], []).append(i)
    # weight: final-1 rows share 0.5, final-3 row 0.5
    w = {i: (0.5 / len(ONE) if FIN[i] == 1 else 0.5 / len(THREE)) for i in ONE + THREE}
    return float(sum(max(sum(w[i] for i in g if FIN[i] == 1), sum(w[i] for i in g if FIN[i] == 3))
                     for g in groups.values()))


def reader_ceilings(add_first_list: list[bool]) -> dict:
    out = {}
    for af in (True, False):
        f = features(af)
        out[str(af)] = {nm: {**{k: ceiling(v, rows) for k, rows in COMP.items()}, "1v3": ceiling_1v3(v)}
                        for nm, v in f.items()}
    return out


# ---------------- shortcut matrices (visual / metadata) ----------------

def _r(o: dict) -> float:
    s = o["size"]
    if o["shape"] in ("sphere", "cylinder"):
        return s[0]
    if o["shape"] == "box":
        return s[0] / 0.8
    raise ValueError(f"unknown shape {o['shape']}")


def _vol(o: dict) -> float:
    s = o["size"]
    if o["shape"] == "sphere":
        return 4 / 3 * math.pi * s[0] ** 3
    if o["shape"] == "box":
        return 8 * s[0] * s[1] * s[2]
    if o["shape"] == "cylinder":
        return 2 * math.pi * s[0] ** 2 * s[1]
    raise ValueError(f"unknown shape {o['shape']}")


def _signed(src: dict, nm: str, f) -> float:
    """Quantity f summed over the box's starting objects, plus/minus f of each object actually carried in/out
    (from the simulated transfer records)."""
    ep = src["episodes"][nm]
    objs = {o["id"]: o for o in ep["objects"]}
    q = sum(f(o) for o in ep["objects"] if o["start_region"] == "box")
    for t in src["transfers"][nm]:
        q += ((t["region_after"] == "box") - (t["region_before"] == "box")) * f(objs[t["object"]])
    return q


def meta_volume_matrix(src: dict, don: dict) -> np.ndarray:
    pred = [_signed(src, nm, _vol) for nm in src["names"]]
    tgt = [sum(_vol(o) for o in don["episodes"][nm]["objects"] if o["start_region"] == "box") for nm in don["names"]]
    return np.abs(np.array(pred)[:, None] - np.array(tgt)[None])


def count_equiv_matrix(src: dict, don: dict) -> np.ndarray:
    """Astra's size-normalised quantity reader (positive control, NOT a shortcut): source area / the carried
    object's area vs donor (R1 / largest radius)^2. Under area matching it recovers the count."""
    from c6_items import DONOR_R1
    pred = []
    for nm in src["names"]:
        ep = src["episodes"][nm]
        objs = {o["id"]: o for o in ep["objects"]}
        unit = math.pi * _r(objs[src["transfers"][nm][0]["object"]]) ** 2
        pred.append(_signed(src, nm, lambda o: math.pi * _r(o) ** 2) / unit)
    tgt = [(DONOR_R1 / max(_r(o) for o in don["episodes"][nm]["objects"] if o["start_region"] == "box")) ** 2
           for nm in don["names"]]
    return np.abs(np.array(pred)[:, None] - np.array(tgt)[None])


def actual_finals(src: dict) -> np.ndarray:
    """Finals from the simulated transfers (independent of the schedule): N + carried in - carried out."""
    return np.array([src["meta"][nm]["n0"] + sum((t["region_after"] == "box") - (t["region_before"] == "box")
                                                 for t in src["transfers"][nm]) for nm in src["names"]])


DIRECTIONS = ("nearest", "farthest", "inverted")


def directed(S: np.ndarray, d: str) -> dict:
    """Endpoints of a shortcut reader under each frozen decision rule: nearest ending, farthest ending, and the
    nearest rule with the binary 'is 2' answer inverted (I-III only; 1v3/exact are reported for nearest only)."""
    if d == "nearest":
        return block_endpoints(S)
    if d == "farthest":
        return block_endpoints(-S)
    e = block_endpoints(S)
    for k in ("I", "II", "III"):
        e[k] = 1 - e[k]
    e["mean_I_II"] = (e["I"] + e["II"]) / 2
    e["mean_I_III"] = (e["I"] + e["II"] + e["III"]) / 3
    return e


def meta_area_matrix(src: dict, don: dict) -> np.ndarray:
    """Initial box-object area + signed area of each object carried in (HB) / out (BH), vs donor ending area
    (sum of pi r^2 of the box objects): the area-tracking reader the area matching is meant to defeat."""
    pred = [_signed(src, nm, lambda o: math.pi * _r(o) ** 2) for nm in src["names"]]
    tgt = [sum(math.pi * _r(o) ** 2 for o in don["episodes"][nm]["objects"] if o["start_region"] == "box")
           for nm in don["names"]]
    return np.abs(np.array(pred)[:, None] - np.array(tgt)[None])


def pixel_matrix(xs: np.ndarray, xd: np.ndarray, ctx: int, first: int, common: np.ndarray, mode: str) -> np.ndarray:
    """L1 over the common box patches between a source frame and the donor's post-reveal frames (mean).
    mode 'last_ctx': the source's last (hopper-masked) context frame; 'first': its frame 0."""
    pm = np.zeros((128 // 8) ** 2, bool); pm[common] = True
    pm = np.kron(pm.reshape(16, 16), np.ones((8, 8), bool))
    s_fr = xs[:, ctx - 1] if mode == "last_ctx" else xs[:, 0]
    d_fr = xd[:, ctx + first * TUB:].mean(1)
    a = s_fr[:, pm].astype(float); b = d_fr[:, pm].astype(float)
    return np.abs(a[:, None] - b[None]).mean((2, 3))


# ---------------- model ----------------

def mask_hopper(fr: np.ndarray, m: np.ndarray, ring: np.ndarray, ctx: int) -> np.ndarray:
    fr = fr.copy()
    fill = np.median(fr[0, 0][ring], 0).astype(fr.dtype)
    fr[:, :ctx][:, :, m] = fill
    return fr


def load_model(run: str, weights: str):
    from jepatrain.reeval import load_run
    if weights != "untrained":
        enc, tgt, pred, step = load_run(run, weights, device="cpu")
        return enc, tgt, pred, step
    from jepatrain.model import Encoder, Predictor
    args = json.load(open(os.path.join(run, "args.json")))
    torch.manual_seed(0)
    mk = lambda: Encoder(img=args["res"], patch=args["patch"], dim=args["dim"], depth=args["depth"], heads=args["dim"] // 64)
    enc = mk(); tgt = mk(); tgt.load_state_dict(enc.state_dict())
    pred = Predictor(enc, dim=args["pred_dim"], depth=args["pred_depth"], heads=args["pred_dim"] // 32)
    return enc, tgt, pred, 0


@torch.no_grad()
def model_matrix(xs: np.ndarray, xd: np.ndarray, ctx: int, first: int, common: np.ndarray, enc, tgt, pred) -> np.ndarray:
    def prep(fr):
        x = torch.from_numpy(fr).permute(0, 4, 1, 2, 3).float() / 255
        return (x - MEAN) / STD
    a, b = prep(xs), prep(xd)
    G = enc.grid * enc.grid
    slots = a.shape[2] // TUB
    s0 = ctx // TUB + first
    tok = torch.from_numpy(np.asarray(common))
    keep = torch.cat([s * G + tok for s in range(s0, slots)])
    n_ctx, N = ctx // TUB * G, slots * G
    z = enc(a[:, :, :ctx])
    P = pred(z, torch.arange(n_ctx).unsqueeze(0).expand(len(xs), -1),
             torch.arange(n_ctx, N).unsqueeze(0).expand(len(xs), -1)).float()[:, keep - n_ctx]
    h = tgt(b, keep.unsqueeze(0).expand(len(xd), -1)).float()
    T = F.layer_norm(h, (h.size(-1),))
    return (P[:, None] - T[None]).abs().mean((2, 3)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--weights", action="append", default=[])
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--boot", type=int, default=10_000)
    a = ap.parse_args()
    if not (len(a.run) == len(a.weights) == len(a.name)):
        raise SystemExit("--run/--weights/--name must pair up")
    man = json.load(open(a.manifest))
    for d_, h in man["items_sha256"].items():
        if hashlib.sha256(open(os.path.join(d_, "items.jsonl"), "rb").read()).hexdigest() != h:
            raise SystemExit(f"{d_}/items.jsonl changed since the manifest was frozen")
    srec = {json.loads(l)["seed"]: json.loads(l) for l in open(os.path.join(man["sources_dir"], "items.jsonl"))}
    drec = {json.loads(l)["seed"]: json.loads(l) for dd in man["donors_dirs"] for l in open(os.path.join(dd, "items.jsonl"))}
    blocks = []
    for p in man["pairs"]:
        sp = os.path.join(man["sources_dir"], f"source_{p['source']:09d}.npz")
        dp = os.path.join(p["donor_dir"], f"donor_{p['donor']:09d}.npz")
        for path, h in ((sp, p["source_sha256"]), (dp, p["donor_sha256"])):
            if hashlib.sha256(open(path, "rb").read()).hexdigest() != h:
                raise SystemExit(f"{path} changed since the manifest was frozen")
        s, d = srec[p["source"]], drec[p["donor"]]
        if s["names"] != [c[0] for c in CELLS] or [int(x) for x in d["names"]] != [1, 2, 3]:
            raise SystemExit(f"block {p['source']}/{p['donor']}: unexpected row/column order")
        if [s["meta"][n]["final"] for n in s["names"]] != FIN.tolist() or actual_finals(s).tolist() != FIN.tolist():
            raise SystemExit(f"source {p['source']}: scheduled {[s['meta'][n]['final'] for n in s['names']]}, "
                             f"simulated {actual_finals(s).tolist()}")
        blocks.append((p, s, d, sp, dp))
    strata = np.array([s["add_first"] for _, s, _, _, _ in blocks])
    res = {"manifest": a.manifest, "n_blocks": len(blocks), "add_first": {str(k): int((strata == k).sum()) for k in (True, False)},
           "reader_ceilings": reader_ceilings(list(strata)), "models": {}}
    # the state counter scores the simulated finals (actual_finals), so a mis-executed transfer would show here
    per = {"state_counter": [block_endpoints(-np.eye(3)[actual_finals(s) - 1]) for _, s, _, _, _ in blocks],
           "count_equiv": []}
    sc = {f"{r}:{d_}": [] for r in ("meta_area", "meta_volume", "pixel_last_ctx", "pixel_first") for d_ in DIRECTIONS}
    for p, s, d, sp, dp in blocks:
        zs, zd = np.load(sp), np.load(dp)
        xs = mask_hopper(zs["frames"], zs["hopper_px"], zs["hopper_ring"], s["ctx"])
        common = np.array(p["common"])
        mats = {"meta_area": meta_area_matrix(s, d), "meta_volume": meta_volume_matrix(s, d),
                "pixel_last_ctx": pixel_matrix(xs, zd["frames"], s["ctx"], s["first_post_slot"], common, "last_ctx"),
                "pixel_first": pixel_matrix(xs, zd["frames"], s["ctx"], s["first_post_slot"], common, "first")}
        for r, M in mats.items():
            for d_ in DIRECTIONS:
                sc[f"{r}:{d_}"].append(directed(M, d_))
        per["count_equiv"].append(block_endpoints(count_equiv_matrix(s, d)))
    per.update(sc)
    torch.set_num_threads(16)
    for run, w, name in zip(a.run, a.weights, a.name):
        enc, tgt, pred, step = load_model(run, w)
        enc.eval(); tgt.eval(); pred.eval()
        mats = []
        for p, s, d, sp, dp in blocks:
            zs, zd = np.load(sp), np.load(dp)
            xs = mask_hopper(zs["frames"], zs["hopper_px"], zs["hopper_ring"], s["ctx"])
            mats.append(model_matrix(xs, zd["frames"], s["ctx"], s["first_post_slot"], np.array(p["common"]), enc, tgt, pred))
        per[name] = [block_endpoints(S) for S in mats]
        res["models"][name] = {"run": run, "weights": w, "step": step, "confusion_true_by_chosen": confusion(mats)}
        np.savez_compressed(os.path.splitext(a.out)[0] + f"_{name}_mats.npz", mats=np.stack(mats),
                            source=np.array([p["source"] for p, *_ in blocks]), donor=np.array([p["donor"] for p, *_ in blocks]))
        print(f"[{name}] " + "  ".join(f"{k} {np.mean([b[k] for b in per[name]]):.3f}" for k in per[name][0]), flush=True)
        del mats
    pairs = [(m, s_) for m in res["models"] for s_ in sc] + \
            [(m1, m2) for m1 in res["models"] for m2 in res["models"] if m1 != m2]
    res["bootstrap"] = bootstrap(per, strata, a.boot, pairs=pairs)
    res["by_order"] = {nm: {str(o): {k: float(np.mean([b[k] for b, st_ in zip(bl, strata) if st_ == o]))
                                     for k in bl[0]} for o in (True, False)} for nm, bl in per.items()}
    for nm in per:
        if nm not in res["models"]:
            print(f"[reader] {nm:16s} " + "  ".join(f"{k} {res['bootstrap'][nm][k]['est']:.3f}" for k in ("I", "II", "III", "1v3", "exact")), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
