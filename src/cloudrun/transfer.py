"""Instance-side B2 transfers with the instance credentials (CLOUDRUN_B2_*).

    python -m cloudrun.transfer down --prefix data/v2_128 --local $DATA/v2_128
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import b2
from .creds import b2_from_env


def main() -> int:
    ap = argparse.ArgumentParser(prog="cloudrun.transfer")
    ap.add_argument("action", choices=("down",))
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--local", required=True)
    ap.add_argument("--allow-empty", action="store_true", help="an empty prefix is fine (first episode)")
    a = ap.parse_args()
    s3 = b2.client(b2_from_env())
    b2.download_prefix(s3, os.environ["CLOUDRUN_BUCKET"], a.prefix.rstrip("/"), Path(a.local),
                       require=not a.allow_empty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
