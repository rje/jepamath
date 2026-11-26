"""Instance side: a liveness pulse, independent of uploads and of the job's phase.

onstart starts this right after the code fetch, before restore, and it runs until the box goes away. It writes
`runs/<job>/pulse/ep<NNN>.json` ({"episode", "time"}) every `interval` seconds. It is a separate object and a
separate process from the syncer's heartbeat, so a long restore, a slow checkpoint upload or the final flush
never makes it look stale. The supervisor replaces a box only when the pulse itself stops changing
(watch.hb_stale_min): the box is frozen, or it has lost its network.

    python -m cloudrun.pulse --interval 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import b2
from .creds import b2_from_env


def key(root: str, job: str, ep: int) -> str:
    return "/".join(p for p in (root.strip("/"), "runs", job, "pulse", f"ep{ep:03d}.json") if p)


def main() -> int:
    ap = argparse.ArgumentParser(prog="cloudrun.pulse")
    ap.add_argument("--interval", type=float, default=60)
    a = ap.parse_args()
    e = os.environ
    s3 = b2.client(b2_from_env())
    k = key(e.get("CLOUDRUN_ROOT", ""), e["CLOUDRUN_JOB"], int(e["CLOUDRUN_EPISODE"]))
    while True:
        try:
            b2.put_bytes(s3, e["CLOUDRUN_BUCKET"], k,
                         json.dumps({"episode": int(e["CLOUDRUN_EPISODE"]), "time": time.time()}).encode())
        except Exception as ex:     # noqa: BLE001 — transient B2 error; a lasting one shows as a stale pulse
            print(f"[pulse] error: {type(ex).__name__}: {ex}", file=sys.stderr, flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
