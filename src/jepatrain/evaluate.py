"""Track B evaluation: reveal-locked twin surprise (same protocol as Track A) and count probes."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from jepaeval.twins import _screen_times, reveal_on_triples
from jepaeval.vjepa import _SurpriseBase

FPS = 4.0


class TrainedSurprise(_SurpriseBase):
    """Wraps a Track B (encoder, target encoder, predictor) in the jepaeval surprise interface."""

    def __init__(self, enc, tgt, pred, guard, precision: str = "bf16", device: str = "cuda"):
        self._setup(device, precision, guard, window_batch=16)
        self.enc, self.tgt, self.pred = enc, tgt, pred
        self.res, self.grid, self.tubelet = enc.img, enc.grid, enc.tubelet

    def _pred_tgt(self, x, c):
        B, T = x.shape[0], x.shape[2]
        G = self.grid * self.grid
        n_ctx, N = c // self.tubelet * G, T // self.tubelet * G
        h = self.tgt(x).float()
        tgt = F.layer_norm(h, (h.size(-1),))[:, n_ctx:]
        z = self.enc(x[:, :, :c])
        ar = lambda a, b: torch.arange(a, b, device=x.device).unsqueeze(0).expand(B, -1)
        p = self.pred(z, ar(0, n_ctx), ar(n_ctx, N))
        return p.float(), tgt

    @torch.no_grad()
    def reveal_features(self, frames: np.ndarray, ep: dict) -> dict:
        """Pooled features of one possible clip: start (encoder on the first tubelet), true post-reveal
        (target encoder on the full clip), predicted post-reveal (predictor from the pre-lowering context)."""
        x = self.preprocess(frames).unsqueeze(0)
        tb, G, T = self.tubelet, self.grid * self.grid, x.shape[2]
        t0, t1 = _screen_times(ep)
        ctx = int(t0 * FPS) // tb * tb
        f_end = int(np.ceil(t1 * FPS))
        first = next(j for j in range((T - ctx) // tb) if ctx + tb * (j + 1) - 1 >= f_end)
        n_ctx, N = ctx // tb * G, T // tb * G
        with self.guard, torch.autocast("cuda", dtype=self.autocast or torch.float16, enabled=self.autocast is not None):
            start = self.enc(x[:, :, :tb]).float().mean(1)
            h = self.tgt(x).float()
            h = F.layer_norm(h, (h.size(-1),))
            ar = lambda a, b: torch.arange(a, b, device=x.device).unsqueeze(0)
            p = self.pred(self.enc(x[:, :, :ctx]), ar(0, n_ctx), ar(n_ctx, N)).float()
        post_true = h[:, n_ctx + first * G:].mean(1)
        post_pred = p[:, first * G:].mean(1)
        return {k: v[0].cpu().numpy() for k, v in
                {"start": start, "post_true": post_true, "post_pred": post_pred}.items()}


def _probe(Xtr, ytr, Xte_list, yte_list):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xtr), ytr)
    return [float((clf.predict(sc.transform(X)) == y).mean() * 100) for X, y in zip(Xte_list, yte_list)]


def count_probes(sj: TrainedSurprise, triples: list[dict]) -> dict:
    """Linear count probes on possible clips; first half trains, second half tests.
      n0@start          : is the starting count decodable from the first tubelet?
      final@post_true   : is the final count decodable when it is visible (target encoder)?
      final@post_pred   : does the predicted reveal carry the right final count (probe from post_true)?
      n0@post_pred      : ... or the unchanged starting count (the model ignored the transfers)?"""
    F_ = [sj.reveal_features(t["possible"], t["pos_episode"]) for t in triples]
    n0 = np.array([t["meta"]["n0"] for t in triples])
    fin = np.array([t["meta"]["final"] for t in triples])
    changed = n0 != fin
    h = len(triples) // 2
    S = np.stack([f["start"] for f in F_])
    PT = np.stack([f["post_true"] for f in F_])
    PP = np.stack([f["post_pred"] for f in F_])
    (a_n0,) = _probe(S[:h], n0[:h], [S[h:]], [n0[h:]])
    a_true, a_pred, a_pred_n0, a_pred_chg = _probe(
        PT[:h], fin[:h], [PT[h:], PP[h:], PP[h:], PP[h:][changed[h:]]],
        [fin[h:], fin[h:], n0[h:], fin[h:][changed[h:]]])
    maj = lambda y: float((y[h:] == np.bincount(y[:h]).argmax()).mean() * 100)
    return {"n0@start": a_n0, "final@post_true": a_true, "final@post_pred": a_pred,
            "n0@post_pred": a_pred_n0, "final@post_pred|changed": a_pred_chg,
            "chance_n0": maj(n0), "chance_final": maj(fin), "n_test": len(triples) - h}


def evaluate_all(sj: TrainedSurprise, triples: list[dict], n_twins: int = 1000) -> dict:
    for m in (sj.enc, sj.tgt, sj.pred):
        m.eval()
    r = reveal_on_triples(triples[:n_twins], sj, FPS, log=lambda *_: None)
    probes = count_probes(sj, triples)
    for m in (sj.enc, sj.pred):
        m.train()
    return {"reveal": r["metrics"], "probes": probes, "raw": r["raw"]}
