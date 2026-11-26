"""python -m cloudrun [--project cloud/cloudrun.toml] <command> ...

  offers JOB              rank live offers for the job's machine spec (read-only)
  plan JOB                every pre-rental check + the top offers; rents nothing
  launch JOB --yes        supervise the job (rents; without --yes it is `plan`)
  status JOB              supervisor state, newest heartbeat, live instance status
  logs JOB [--ep N]       the shipped log tail of an episode (default: newest)
  pull JOB DEST           download the job's run dir from B2
  push-data LOCAL REMOTE  upload a directory to the bucket (resumable, verified)
  ls PREFIX               list bucket objects
  instances               every instance on the Vast account
  destroy ID              destroy one instance (verified)
  make-bucket             create the project bucket (private, old versions deleted after 1 day)
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import b2
from .config import load_job, load_project
from .creds import b2_from_file, read_env_file, vast_key
from .supervisor import State, data_gb, heartbeat, run, recover
from .vast import Vast, fmt_offers, rank


def _git(*args) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def code_commit() -> str:
    """HEAD, which must be clean and pushed: the instance fetches exactly this commit from GitHub."""
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise SystemExit(f"working tree has uncommitted changes; commit and push first:\n{dirty}")
    head = _git("rev-parse", "HEAD")
    if not _git("branch", "-r", "--contains", head):
        raise SystemExit(f"HEAD {head[:8]} is not on any remote branch; git push first")
    return head


def checks(job, project, s3) -> list[str]:
    msgs = []
    for d in job.data:
        have = b2.remote_sizes(s3, project.bucket, project.key(d.remote) + "/")
        if not have:
            raise SystemExit(f"data s3://{project.bucket}/{project.key(d.remote)}/ is empty — push-data first")
        msgs.append(f"data {d.remote}: {len(have)} objects, {b2.human(sum(have.values()))}")
    return msgs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cloudrun", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="cloud/cloudrun.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("offers", "plan", "status"):
        sub.add_parser(c).add_argument("job")
    p = sub.add_parser("launch"); p.add_argument("job"); p.add_argument("--yes", action="store_true")
    p = sub.add_parser("logs"); p.add_argument("job"); p.add_argument("--ep", type=int)
    p = sub.add_parser("pull"); p.add_argument("job"); p.add_argument("dest")
    p = sub.add_parser("push-data"); p.add_argument("local"); p.add_argument("remote")
    p = sub.add_parser("ls"); p.add_argument("prefix")
    sub.add_parser("instances")
    p = sub.add_parser("destroy"); p.add_argument("id", type=int)
    sub.add_parser("make-bucket")
    a = ap.parse_args(argv)

    project = load_project(a.project)
    creds = b2_from_file(project.b2_env)
    s3 = b2.client(creds)
    jobfile = lambda name: Path(a.project).parent / "jobs" / f"{name}.toml"

    if a.cmd in ("offers", "plan", "launch"):
        job = load_job(jobfile(a.job))
        vast = Vast(vast_key(project.vast_env))
        if a.cmd == "launch":
            # close anything a previous supervisor left billing before any precheck can fail
            st = State(project.path(project.state_dir) / f"{job.name}.json", job.name)
            prior = recover(st, vast, lambda m: print(m, flush=True))
            if prior:
                print(f"{job.name}: {prior} (decided before a previous supervisor's destroy failed)")
                return 0 if prior.startswith("done") else 1
        down_gb = data_gb(job, project, s3)
        ranked = rank(vast.search_all(job.machine), job.machine, down_gb)
        print(f"{len(ranked)} eligible offers for {job.machine.gpus} x{job.machine.num_gpus} "
              f"at <= ${job.machine.max_price_per_gpu}/GPU-h; ranked by expected episode $ "
              f"({job.machine.episode_hours} h + {down_gb:.1f} GB in + {job.machine.upload_gb} GB out)")
        print(fmt_offers(ranked, m=job.machine, down_gb=down_gb))
        if a.cmd == "offers":
            return 0
        commit = code_commit()
        for m in checks(job, project, s3):
            print(m)
        inst_creds = b2_from_file(project.instance_b2_env or project.b2_env)
        if not ranked:
            raise SystemExit("no eligible offers")
        rate = ranked[0][0]
        print(f"\nPLAN {job.name} @ {commit[:8]}\n  cheapest rate ${rate:.3f}/h; hard caps ${job.budget.max_spend:.2f} "
              f"and {job.budget.max_hours} h (≈ ${rate * job.budget.max_hours:.2f} if it ran the full wall clock)\n"
              f"  command: {job.command}\n  results -> s3://{project.bucket}/{project.key('runs', job.name, 'run')}/")
        if a.cmd == "plan" or not a.yes:
            print("\ndry run: nothing rented. `launch JOB --yes` rents.")
            return 0
        outcome = run(job, project, vast, s3, inst_creds, commit, out=lambda m: print(m, flush=True))
        return 0 if outcome == "done" else 1

    if a.cmd == "status":
        st = State(project.path(project.state_dir) / f"{a.job}.json", a.job)
        print(json.dumps(st.d, indent=1))
        if st.d["episodes"]:
            e = st.d["episodes"][-1]
            print("heartbeat:", json.dumps(heartbeat(s3, project, a.job, e["n"])))
            if e["end"] is None:
                print("instance status:", Vast(vast_key(project.vast_env)).status(e["instance"]))
        return 0
    if a.cmd == "logs":
        keys = sorted(b2.remote_sizes(s3, project.bucket, project.key("runs", a.job, "logs") + "/"))
        if not keys:
            raise SystemExit("no logs shipped yet")
        key = project.key("runs", a.job, "logs", f"ep{a.ep:03d}.log") if a.ep is not None else keys[-1]
        r = b2.get_bytes(s3, project.bucket, key)
        if r is None:
            raise SystemExit(f"no {key}")
        print(f"--- {key} (modified {time.ctime(r[1])})")
        sys.stdout.write(r[0].decode("utf-8", "replace"))
        return 0
    if a.cmd == "pull":
        b2.download_prefix(s3, project.bucket, project.key("runs", a.job, "run"), Path(a.dest))
        return 0
    if a.cmd == "push-data":
        b2.upload_dir(s3, Path(a.local), project.bucket, project.key(a.remote))
        return 0
    if a.cmd == "ls":
        have = b2.remote_sizes(s3, project.bucket, project.key(a.prefix))
        for k, v in sorted(have.items()):
            print(f"{v:>14,}  {k}")
        print(f"{len(have)} objects, {b2.human(sum(have.values()))}")
        return 0
    if a.cmd == "instances":
        for i in Vast(vast_key(project.vast_env)).instances():
            print(f"{i.get('id')}  {i.get('actual_status')}  ${i.get('dph_total', 0):.3f}/h  {i.get('gpu_name')}  "
                  f"{i.get('label') or ''}")
        return 0
    if a.cmd == "destroy":
        Vast(vast_key(project.vast_env)).destroy(a.id)
        print(f"destroyed {a.id} (verified)")
        return 0
    if a.cmd == "make-bucket":
        return make_bucket(project)
    raise AssertionError(a.cmd)


def make_bucket(project) -> int:
    """Private bucket whose lifecycle rule deletes superseded versions after a day. B2 keeps every version of
    an overwritten key, and the syncer overwrites heartbeats and log tails every minute."""
    d = read_env_file(project.b2_env)
    key = d.get("B2_APPLICATION_KEY") or d.get("B2_APP_KEY")
    auth = base64.b64encode(f"{d['B2_KEY_ID']}:{key}".encode()).decode()
    acct = json.load(urllib.request.urlopen(urllib.request.Request(
        "https://api.backblazeb2.com/b2api/v3/b2_authorize_account", headers={"Authorization": "Basic " + auth})))
    api = acct["apiInfo"]["storageApi"]
    body = {"accountId": acct["accountId"], "bucketName": project.bucket, "bucketType": "allPrivate",
            "lifecycleRules": [{"fileNamePrefix": "", "daysFromHidingToDeleting": 1,
                                "daysFromUploadingToHiding": None}]}
    req = urllib.request.Request(f"{api['apiUrl']}/b2api/v3/b2_create_bucket", data=json.dumps(body).encode(),
                                 headers={"Authorization": acct["authorizationToken"]}, method="POST")
    try:
        r = json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"create bucket failed: {e.code} {e.read().decode()[:300]}")
    print(f"created bucket {r['bucketName']} ({r['bucketType']}), lifecycle {r['lifecycleRules']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
