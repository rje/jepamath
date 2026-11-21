"""Credentials: env files (mode 600) on the local machine; prefixed environment variables on an instance.

Secrets never go on a command line (argv is world-readable via /proc) and are never printed.
"""

from __future__ import annotations

import dataclasses as dc
import os
from pathlib import Path

INSTANCE_PREFIX = "CLOUDRUN_B2"


def read_env_file(path: str | Path) -> dict[str, str]:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"credential file {p} not found")
    mode = p.stat().st_mode & 0o777
    if mode & 0o077:
        raise PermissionError(f"{p} is mode {mode:o}; it holds a secret. Run: chmod 600 {p}")
    out = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


@dc.dataclass(frozen=True)
class B2Creds:
    key_id: str
    app_key: str
    endpoint: str

    def __repr__(self):     # never leak the secret into a log or traceback
        return f"B2Creds(key_id={self.key_id[:6]}…, endpoint={self.endpoint})"

    def as_instance_env(self) -> dict[str, str]:
        return {f"{INSTANCE_PREFIX}_KEY_ID": self.key_id, f"{INSTANCE_PREFIX}_APPLICATION_KEY": self.app_key,
                f"{INSTANCE_PREFIX}_ENDPOINT": self.endpoint}


def b2_from_file(path: str | Path) -> B2Creds:
    d = read_env_file(path)
    # the B2 console calls the secret "applicationKey"; microlab's files use both spellings
    key = d.get("B2_APPLICATION_KEY") or d.get("B2_APP_KEY")
    missing = [n for n, v in (("B2_KEY_ID", d.get("B2_KEY_ID")), ("B2_APPLICATION_KEY", key),
                              ("B2_ENDPOINT", d.get("B2_ENDPOINT"))) if not v]
    if missing:
        raise KeyError(f"{path}: missing {missing}")
    return B2Creds(d["B2_KEY_ID"], key, d["B2_ENDPOINT"])


def b2_from_env(prefix: str = INSTANCE_PREFIX) -> B2Creds:
    vals = [os.environ.get(f"{prefix}_{k}", "") for k in ("KEY_ID", "APPLICATION_KEY", "ENDPOINT")]
    if not all(vals):
        raise KeyError(f"instance credentials {prefix}_KEY_ID/_APPLICATION_KEY/_ENDPOINT not all set")
    return B2Creds(*vals)


def vast_key(path: str | Path) -> str:
    d = read_env_file(path)
    if not d.get("VAST_API_KEY"):
        raise KeyError(f"{path}: no VAST_API_KEY")
    return d["VAST_API_KEY"]
