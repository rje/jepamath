"""Figures for the research log from a run's log.jsonl."""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(run: str):
    tr, ev = [], []
    with open(os.path.join(run, "log.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            (ev if "eval_step" in r else tr).append(r)
    return tr, ev


def plot_run(run: str, out: str, title: str | None = None):
    tr, ev = load(run)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    s = [r["step"] for r in tr]
    for k, c in (("loss_prefix", "C0"), ("loss_short", "C1"), ("loss_long", "C2")):
        pts = [(r["step"], r[k]) for r in tr if k in r]
        if pts:
            ax[0].plot(*zip(*pts), c, lw=1, label=k.replace("loss_", ""))
    ax[0].set(title="training loss (L1, latent)", xlabel="step"); ax[0].legend()
    ax0b = ax[0].twinx()
    ax0b.plot(s, [r["tgt_std"] for r in tr], "k:", lw=1, label="target std")
    ax0b.set_ylabel("target std (collapse check)")
    if ev:
        es = [r["eval_step"] for r in ev]
        g = lambda r, grp, cmp: (r["reveal"]["post_reveal"]["by"][grp][cmp]["acc"] if grp
                                 else r["reveal"]["post_reveal"][cmp]["acc"])
        for lab, grp, cmp, sty in (("imp>ctl (all)", None, "imp>ctl", "k-o"), ("imp>ctl | add", "imp=add", "imp>ctl", "C3-o"),
                                   ("imp>ctl | delete", "imp=delete", "imp>ctl", "C0-o"), ("ctl>pos", None, "ctl>pos", "C7--")):
            ax[1].plot(es, [g(r, grp, cmp) for r in ev], sty, ms=3, label=lab)
        ax[1].axhline(50, c="grey", lw=0.8)
        ax[1].set(title="reveal-locked surprise, 1000 held-out twins", xlabel="step", ylabel="% seeds", ylim=(0, 100))
        ax[1].legend(fontsize=8)
        for k, sty in (("n0@start", "C2-o"), ("final@post_true", "C1-o"), ("final@post_pred", "C3-o"),
                       ("n0@post_pred", "C0--"), ("final@post_pred|changed", "C4-o")):
            ax[2].plot(es, [r["probes"][k] for r in ev], sty, ms=3, label=k)
        ax[2].axhline(ev[-1]["probes"]["chance_final"], c="grey", lw=0.8, ls=":", label="majority (final)")
        ax[2].set(title="linear count probes (held-out)", xlabel="step", ylabel="accuracy %", ylim=(0, 100))
        ax[2].legend(fontsize=8)
    fig.suptitle(title or os.path.basename(run.rstrip("/")))
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    plot_run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
