"""jepagen command line."""

from __future__ import annotations

import argparse
import json
import sys

from .sampler import Manifest, Template, load_yaml


def _common(p):
    p.add_argument("--template", default=None, help="template YAML (default: built-in)")
    p.add_argument("--manifest", default=None, help="split manifest YAML (default: built-in)")
    p.add_argument("--split", default="train")
    p.add_argument("--backend", default="mujoco", choices=["mujoco", "rust"])
    p.add_argument("--max-render-fps", type=float, default=60.0,
                   help="TOTAL GPU renders per second across all workers")


def _load(args):
    t = load_yaml(args.template, Template) if args.template else Template()
    m = load_yaml(args.manifest, Manifest) if args.manifest else Manifest()
    return t, m


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jepagen")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    _common(g)
    g.add_argument("--out", required=True)
    g.add_argument("--seeds", type=int, nargs=2, required=True, metavar=("START", "STOP"))
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--twins", action="store_true", help="also generate impossible twins (separate shards)")
    g.add_argument("--temp-ceiling", type=float, default=83.0)

    pv = sub.add_parser("preview")
    _common(pv)
    pv.add_argument("--out", required=True)
    pv.add_argument("--seeds", type=int, nargs=2, default=(0, 20), metavar=("START", "STOP"))

    au = sub.add_parser("audit")
    au.add_argument("--data", required=True)
    au.add_argument("--split", default="train")
    au.add_argument("--max-clips", type=int, default=10_000)
    au.add_argument("--margin", type=float, default=0.10)

    cc = sub.add_parser("crosscheck")
    _common(cc)
    cc.add_argument("--out", required=True)
    cc.add_argument("--seeds", type=int, nargs=2, default=(0, 20), metavar=("START", "STOP"))

    sub.add_parser("schema")

    args = ap.parse_args(argv)
    if args.cmd == "generate":
        from .generate import generate
        t, m = _load(args)
        print(json.dumps(generate(t, m, args.split, range(*args.seeds), args.out, args.backend, args.twins,
                                  args.workers, args.max_render_fps, args.temp_ceiling)))
    elif args.cmd == "preview":
        from .preview import preview
        t, m = _load(args)
        for line in preview(t, m, args.split, range(*args.seeds), args.out, args.backend, args.max_render_fps):
            print(line)
    elif args.cmd == "audit":
        from .audit import audit
        res = audit(args.data, args.split, args.max_clips, args.margin)
        print(json.dumps(res, indent=1))
        sys.exit(1 if res["leak"] else 0)
    elif args.cmd == "crosscheck":
        from .crosscheck import crosscheck
        t, m = _load(args)
        rows = crosscheck(t, m, args.split, range(*args.seeds), args.out, max_render_fps=args.max_render_fps)
        bad = [r for r in rows if r["count_frames_mismatched"] or not r["events_match"]]
        for r in rows:
            print(json.dumps(r))
        print(f"{len(rows) - len(bad)}/{len(rows)} episodes semantically identical across backends")
        sys.exit(1 if bad else 0)
    elif args.cmd == "schema":
        from .spec import json_schema
        print(json.dumps(json_schema(), indent=1))


if __name__ == "__main__":
    main()
