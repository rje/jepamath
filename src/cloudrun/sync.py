"""Instance side: mirror the run directory to B2 and publish a heartbeat, every `interval` seconds.

    python -m cloudrun.sync --run $RUN_DIR --log $WORK/job.log [--once]

Environment (set by the supervisor): CLOUDRUN_BUCKET, CLOUDRUN_JOB, CLOUDRUN_EPISODE, CLOUDRUN_PROGRESS_RE,
CLOUDRUN_ROLLING, CLOUDRUN_KEEP_ROLLING, CLOUDRUN_ROOT (bucket key prefix), CLOUDRUN_B2_* credentials.

B2 layout under [<root>/]runs/<job>/:
  run/<relpath>           the run directory (checkpoints, logs, eval outputs)
  logs/ep<NNN>.log        tail of this episode's job log
  heartbeat/ep<NNN>.json  {"episode", "time", "phase", "step", "exit_rc", "synced"}; exit_rc only from --final

Keys are per episode, so nothing a dead box shipped can be read as the live box's progress.
A local file is uploaded when its (size, mtime) changed and it has been stable for a few seconds: a
checkpoint still being written is never shipped half-done. The trainer writes checkpoints atomically
(tmp + rename) as well, so both ends guard it.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
import time
from pathlib import Path

from . import b2
from .creds import b2_from_env

PHASE = re.compile(r"^CLOUDRUN_PHASE (\S+)", re.M)
EXIT = re.compile(r"^CLOUDRUN_EXIT rc=(-?\d+)", re.M)
STABLE_S = 3.0


def parse_log(text: str, progress_re: re.Pattern) -> dict:
    """phase = last CLOUDRUN_PHASE marker, step = highest progress match, exit_rc = the EXIT sentinel."""
    phases = PHASE.findall(text)
    steps = [int(m.group(1)) for m in progress_re.finditer(text)]
    ex = EXIT.findall(text)
    return {"phase": phases[-1] if phases else "boot", "step": max(steps) if steps else 0,
            "exit_rc": int(ex[-1]) if ex else None}


def rolling_to_prune(keys: list[str], pattern: str, keep: int) -> list[str]:
    """Remote run-dir keys matching `pattern` beyond the newest `keep` (by name: zero-padded steps sort)."""
    if keep <= 0:
        return []
    hits = sorted(k for k in keys if fnmatch.fnmatch(k.rsplit("/", 1)[-1], pattern))
    return hits[:-keep]


class Syncer:
    def __init__(self, s3, bucket: str, job: str, episode: int, run: Path, log: Path, progress_re: str,
                 rolling: str, keep_rolling: int, root: str = "", out=print):
        self.s3, self.bucket, self.job, self.ep = s3, bucket, job, episode
        self.run, self.log, self.re = run, log, re.compile(progress_re)
        self.rolling, self.keep = rolling, keep_rolling
        self.out = out
        self.sent: dict[str, tuple[int, int]] = {}
        self.base = "/".join(p for p in (root.strip("/"), "runs", job) if p)
        self.max_step = 0       # steps seen in earlier passes: the log tail we read can drop old lines

    def seed_from_remote(self) -> None:
        """Files restored from B2 at boot are already there; don't re-upload them."""
        for k, size in b2.remote_sizes(self.s3, self.bucket, f"{self.base}/run/").items():
            p = self.run / k[len(f"{self.base}/run/"):]
            if p.exists() and p.stat().st_size == size:
                st = p.stat()
                self.sent[str(p)] = (st.st_size, st.st_mtime_ns)

    def _changed_files(self) -> list[Path]:
        out = []
        for p in sorted(self.run.rglob("*")) if self.run.exists() else []:
            if not p.is_file() or p.name.endswith((".tmp", ".part")):
                continue
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            if self.sent.get(str(p)) != (st.st_size, st.st_mtime_ns):
                out.append(p)
        return out

    def sync_run_dir(self, stable_s: float = STABLE_S) -> int:
        n = 0
        # newest first, so a preemption mid-pass loses the oldest pending file rather than the newest checkpoint
        for p in sorted(self._changed_files(), key=lambda q: -q.stat().st_mtime if q.exists() else 0):
            try:
                st = p.stat()
                if time.time() - st.st_mtime < stable_s:
                    continue
                key = f"{self.base}/run/{p.relative_to(self.run)}"
                self.s3.upload_file(str(p), self.bucket, key)
                st2 = p.stat()
                if (st2.st_size, st2.st_mtime_ns) != (st.st_size, st.st_mtime_ns):
                    continue        # changed during upload: ship again next pass
                self.sent[str(p)] = (st.st_size, st.st_mtime_ns)
                n += 1
            except FileNotFoundError:
                continue            # the job pruned it between listing and upload
        keys = list(b2.remote_sizes(self.s3, self.bucket, f"{self.base}/run/"))
        for k in rolling_to_prune(keys, self.rolling, self.keep):
            self.s3.delete_object(Bucket=self.bucket, Key=k)
        return n

    def heartbeat(self, synced: int, final: bool = False) -> dict:
        """exit_rc is published only by the final pass (bootstrap's finish, after the job exited): the background
        syncer can see the CLOUDRUN_EXIT line before the final flush has shipped the last files, and the
        supervisor destroys the box as soon as it reads an exit status."""
        text = self.log.read_bytes()[-2_000_000:].decode("utf-8", "replace") if self.log.exists() else ""
        hb = {"episode": self.ep, "time": time.time(), **parse_log(text, self.re), "synced": synced}
        if not final:
            hb["exit_rc"] = None
        hb["step"] = self.max_step = max(self.max_step, hb["step"])
        b2.put_bytes(self.s3, self.bucket, f"{self.base}/logs/ep{self.ep:03d}.log", text[-256_000:].encode())
        b2.put_bytes(self.s3, self.bucket, f"{self.base}/heartbeat/ep{self.ep:03d}.json", json.dumps(hb).encode())
        return hb

    def once(self, final: bool = False) -> dict:
        """final: the job has exited, so nothing is mid-write — ship even files modified a moment ago."""
        n = self.sync_run_dir(0.0 if final else STABLE_S)
        return self.heartbeat(n, final=final)


def main() -> int:
    ap = argparse.ArgumentParser(prog="cloudrun.sync")
    ap.add_argument("--run", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--final", action="store_true", help="with --once: the job has exited; skip the stability wait")
    a = ap.parse_args()
    e = os.environ
    s = Syncer(b2.client(b2_from_env()), e["CLOUDRUN_BUCKET"], e["CLOUDRUN_JOB"], int(e["CLOUDRUN_EPISODE"]),
               Path(a.run), Path(a.log), e["CLOUDRUN_PROGRESS_RE"], e.get("CLOUDRUN_ROLLING", "ckpt_*.pt"),
               int(e.get("CLOUDRUN_KEEP_ROLLING", "2")), root=e.get("CLOUDRUN_ROOT", ""))
    s.seed_from_remote()
    while True:
        try:
            hb = s.once(final=a.final)
            print(f"[sync] {json.dumps(hb)}", flush=True)
        except Exception as ex:     # noqa: BLE001 — a transient B2 error must not end syncing; the next pass retries
            print(f"[sync] error (retrying next pass): {type(ex).__name__}: {ex}", file=sys.stderr, flush=True)
            if a.once:
                raise
        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
