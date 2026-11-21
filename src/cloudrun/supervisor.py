"""Local supervisor for one job: rent, watch, replace, stop, ALWAYS destroy.

Progress is judged only from what reaches B2 (the per-episode heartbeat written by the instance's
syncer), because that is the only signal that survives the instance vanishing. Decisions are made by
`decide()`, a pure function, so every rule here is unit-tested without renting anything.

Rules (in order):
  spend >= budget.max_spend (banked + the running episode)     -> stop
  supervisor wall clock > budget.max_hours                     -> stop
  heartbeat exit_rc == 0                                       -> done
  heartbeat exit_rc != 0, and no episode ever made progress    -> stop (it would fail again on a new box)
  heartbeat exit_rc != 0 otherwise                             -> replace (counts as a failure)
  instance not listed / not in a running state                 -> replace (preemption; not a failure)
  started this episode, step unchanged for watch.stall_min     -> replace (failure: a wedged box bills too)
  not started, episode older than watch.setup_grace_min        -> replace (failure)
  failures > watch.max_failures                                -> stop
"""

from __future__ import annotations

import base64
import dataclasses as dc
import json
import signal
import time
from pathlib import Path

from . import b2
from .config import Job, Project
from .creds import B2Creds
from .vast import RUNNING_STATES, Vast, VastError, fmt_offers, rank


@dc.dataclass
class Obs:
    now: float
    t_start: float              # supervisor start
    ep_start: float
    status: str | None          # instance actual_status, None if not listed
    hb: dict | None             # this episode's heartbeat
    ep_step: int                # highest step seen this episode
    t_step: float               # when ep_step last increased (ep_start if never)
    spent: float                # banked + running episode
    any_progress: bool          # some episode (including this one) reached step > 0
    failures: int


def decide(o: Obs, job: Job) -> tuple[str, str]:
    """-> (action, reason); action in continue | done | replace | replace_failure | stop."""
    b, w = job.budget, job.watch
    if o.spent >= b.max_spend:
        return "stop", f"spend cap: ${o.spent:.2f} >= ${b.max_spend:.2f}"
    if (o.now - o.t_start) / 3600 > b.max_hours:
        return "stop", f"wall-clock cap {b.max_hours} h"
    if o.hb and o.hb.get("exit_rc") is not None:
        rc = o.hb["exit_rc"]
        if rc == 0:
            return "done", "job exited 0"
        if not o.any_progress:
            return "stop", f"job failed (rc={rc}, phase {o.hb.get('phase')}) before any progress"
        return "replace_failure", f"job failed rc={rc} in phase {o.hb.get('phase')}"
    if o.status is None or o.status not in RUNNING_STATES:
        return "replace", f"instance {o.status or 'gone'} (preempted?)"
    if o.ep_step > 0:
        if (o.now - o.t_step) / 60 > w.stall_min:
            return "replace_failure", f"stalled: step {o.ep_step} unchanged for {w.stall_min} min"
    elif (o.now - o.ep_start) / 60 > w.setup_grace_min:
        phase = (o.hb or {}).get("phase", "no heartbeat")
        return "replace_failure", f"no progress {w.setup_grace_min} min into the episode (phase {phase})"
    return "continue", ""


def onstart_script() -> str:
    """What every rented box runs at boot: fetch the code at the commit, then hand over to bootstrap.sh.
    The code fetch retries (container networking can come up after the container does) and a final failure
    is reported to B2, so a broken box is visible from outside rather than an idle bill."""
    return r"""#!/bin/bash
set -uo pipefail
export WORK=/workspace
mkdir -p $WORK && cd $WORK
exec >> $WORK/job.log 2>&1
echo "CLOUDRUN_PHASE fetch  $(date -u +%FT%TZ)"
for i in 1 2 3 4 5; do pip install -q boto3 && break; sleep 10; done
ok=0
for i in 1 2 3 4 5 6 7 8 9 10; do
  python - <<'PY' && { ok=1; break; }
import io, os, shutil, tarfile, urllib.request
url = f"https://codeload.github.com/{os.environ['CLOUDRUN_REPO']}/tar.gz/{os.environ['CLOUDRUN_COMMIT']}"
data = urllib.request.urlopen(url, timeout=120).read()
shutil.rmtree("/workspace/code", ignore_errors=True)
shutil.rmtree("/workspace/code-tmp", ignore_errors=True)
tarfile.open(fileobj=io.BytesIO(data)).extractall("/workspace/code-tmp")
(root,) = os.listdir("/workspace/code-tmp")
os.rename(f"/workspace/code-tmp/{root}", "/workspace/code")
print("code fetched:", url)
PY
  echo "code fetch attempt $i failed"; sleep 15
done
if [ $ok = 0 ]; then
  echo "CLOUDRUN_EXIT rc=94"
  python - <<'PY'
import boto3, json, os, time
e = os.environ
s3 = boto3.client("s3", endpoint_url=e["CLOUDRUN_B2_ENDPOINT"], aws_access_key_id=e["CLOUDRUN_B2_KEY_ID"],
                  aws_secret_access_key=e["CLOUDRUN_B2_APPLICATION_KEY"])
ep = int(e["CLOUDRUN_EPISODE"])
root = e.get("CLOUDRUN_ROOT", "").strip("/")
key = "/".join(p for p in (root, "runs", e["CLOUDRUN_JOB"], "heartbeat", f"ep{ep:03d}.json") if p)
s3.put_object(Bucket=e["CLOUDRUN_BUCKET"], Key=key,
              Body=json.dumps({"episode": ep, "time": time.time(), "phase": "fetch", "step": 0, "exit_rc": 94}))
PY
  exit 94
fi
exec bash $WORK/code/src/cloudrun/bootstrap.sh
"""


def instance_env(job: Job, project: Project, creds: B2Creds, commit: str, episode: int) -> dict[str, str]:
    enc = lambda s: base64.b64encode(s.encode()).decode()
    env = {"CLOUDRUN_JOB": job.name, "CLOUDRUN_EPISODE": str(episode), "CLOUDRUN_BUCKET": project.bucket,
           "CLOUDRUN_REPO": project.repo, "CLOUDRUN_COMMIT": commit,
           "CLOUDRUN_COMMAND_B64": enc(job.command), "CLOUDRUN_SETUP_B64": enc("\n".join(job.setup)),
           "CLOUDRUN_DATA_B64": enc("\n".join(f"{project.key(d.remote)} {d.local}" for d in job.data)),
           "CLOUDRUN_ROOT": project.root,
           "CLOUDRUN_PROGRESS_RE_B64": enc(job.progress_regex),
           "CLOUDRUN_ROLLING": job.rolling, "CLOUDRUN_KEEP_ROLLING": str(job.keep_rolling),
           **creds.as_instance_env()}
    clash = set(env) & set(job.env)
    if clash:
        raise ValueError(f"job.env may not override {sorted(clash)}")
    env.update({k: str(v) for k, v in job.env.items()})
    return env


class State:
    def __init__(self, path: Path, job: str):
        self.path = path
        self.d = json.loads(path.read_text()) if path.exists() else \
            {"job": job, "spent": 0.0, "episodes": [], "failures": 0, "outcome": None}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1))
        tmp.replace(self.path)


def heartbeat(s3, project: Project, job: str, ep: int) -> dict | None:
    r = b2.get_bytes(s3, project.bucket, project.key("runs", job, "heartbeat", f"ep{ep:03d}.json"))
    return json.loads(r[0]) if r else None


class _Terminate(Exception):
    pass


def run(job: Job, project: Project, vast: Vast, s3, creds: B2Creds, commit: str, out=print) -> str:
    """Supervise until done/stop. Returns the outcome string. The current instance is destroyed on every exit
    path, including SIGTERM/SIGHUP (converted to an exception so `finally` runs)."""
    st = State(project.path(project.state_dir) / f"{job.name}.json", job.name)
    if st.d.get("outcome") == "done":
        raise RuntimeError(f"{job.name} is already done (state {st.path}); use a new job name to rerun")
    st.d["outcome"] = None
    any_progress = any(e.get("max_step", 0) > 0 for e in st.d["episodes"])
    # an episode left open by a supervisor that died: its box may still be billing
    for e in st.d["episodes"]:
        if e["end"] is None:
            out(f"episode {e['n']} (instance {e['instance']}) was left open by a previous supervisor")
            _end_episode(st, e, vast, e["instance"], "orphaned by a previous supervisor", out)

    def _sig(signum, _frame):
        raise _Terminate(f"signal {signum}")
    for s in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, _sig)

    t_start = time.time()
    inst = ep = None
    outcome = "stopped"
    try:
        while True:
            if inst is None:
                if st.d["failures"] > job.watch.max_failures:
                    outcome = f"stop: {st.d['failures']} failures > max_failures {job.watch.max_failures}"
                    break
                if st.d["spent"] >= job.budget.max_spend:
                    outcome = f"stop: spend cap ${st.d['spent']:.2f}"
                    break
                ep = len(st.d["episodes"])
                inst, rate, od, offer = provision(job, project, vast, creds, commit, ep, out,
                                                  data_gb(job, project, s3))
                e = {"n": ep, "instance": inst, "rate": rate, "on_demand": od, "offer": offer["id"],
                     "gpu": offer.get("gpu_name"), "geo": offer.get("geolocation"), "start": time.time(),
                     "end": None, "end_reason": None, "max_step": 0, "commit": commit}
                st.d["episodes"].append(e)
                st.save()
                ep_step, t_step = 0, e["start"]
            time.sleep(job.watch.poll_s)
            now = time.time()
            try:
                status, billed, bw = vast.poll(inst)
            except VastError as ex:          # an API hiccup is not evidence the box died; look again next poll
                out(f"  vast API error (ignored this poll): {ex}")
                continue
            if billed is not None and billed > e["rate"]:   # account at what Vast bills, never below the bid
                e["rate"] = billed
            e["bandwidth"] = max(e.get("bandwidth", 0.0), bw)
            hb = heartbeat(s3, project, job.name, ep)
            if hb and hb.get("step", 0) > ep_step:
                ep_step, t_step = hb["step"], now
                e["max_step"] = ep_step
                any_progress = True
            running = e["rate"] * (now - e["start"]) / 3600 + e.get("bandwidth", 0.0)
            o = Obs(now=now, t_start=t_start, ep_start=e["start"], status=status, hb=hb, ep_step=ep_step,
                    t_step=t_step, spent=st.d["spent"] + running, any_progress=any_progress,
                    failures=st.d["failures"])
            action, reason = decide(o, job)
            phase = (hb or {}).get("phase", "-")
            out(f"  [ep{ep} {(now - e['start']) / 60:5.0f}m] {status or 'gone':<8} phase {phase:<8} step {ep_step:>7} "
                f"${o.spent:7.2f}/{job.budget.max_spend:.2f}" + (f"  -> {action}: {reason}" if reason else ""))
            st.save()
            if action == "continue":
                continue
            _end_episode(st, e, vast, inst, reason, out)
            inst = None
            if action in ("done", "stop"):
                outcome = f"{action}: {reason}"
                break
            if action == "replace_failure":
                st.d["failures"] += 1
            st.save()
    except _Terminate as ex:
        outcome = f"stopped: {ex}"
    except KeyboardInterrupt:
        outcome = "stopped: interrupted"
    finally:
        if inst is not None:
            _end_episode(st, e, vast, inst, outcome, out)
        st.d["outcome"] = "done" if outcome.startswith("done") else outcome
        st.save()
        try:
            left = [i for i in vast.instances() if i.get("actual_status") in RUNNING_STATES]
        except VastError as ex:
            left = None
            out(f"could not list instances ({ex}) — CHECK https://cloud.vast.ai/instances/ BY HAND")
        out(f"{job.name}: {outcome}; total spend ~${st.d['spent']:.2f}")
        if left:
            out(f"WARNING: {len(left)} instance(s) still running on the account: {[i.get('id') for i in left]} "
                f"— https://cloud.vast.ai/instances/")
        elif left == []:
            out("confirmed: no running instances on the account")
    return st.d["outcome"]


def _end_episode(st: State, e: dict, vast: Vast, inst: int, reason: str, out) -> None:
    out(f"  destroying {inst} ({reason})")
    try:
        vast.destroy(inst)
        out("  destroyed (verified)")
    except VastError as ex:
        out(f"  DESTROY FAILED: {ex}")
    if e["end"] is None:
        e["end"] = time.time()
        e["end_reason"] = reason
        st.d["spent"] += e["rate"] * (e["end"] - e["start"]) / 3600 + e.get("bandwidth", 0.0)
    st.save()


def data_gb(job, project, s3) -> float:
    """GB a fresh box pulls: the job's data plus its run dir (restore)."""
    prefixes = [project.key(d.remote) + "/" for d in job.data] + [project.key("runs", job.name, "run") + "/"]
    return sum(sum(b2.remote_sizes(s3, project.bucket, p).values()) for p in prefixes) / 1e9


def provision(job: Job, project: Project, vast: Vast, creds: B2Creds, commit: str, ep: int, out,
              down_gb: float = 0.0):
    m = job.machine
    ranked = rank(vast.search_all(m), m, down_gb)
    if not ranked:
        raise RuntimeError(f"no offer for {m.gpus} at <= ${m.max_price_per_gpu}/GPU-h passes the filters")
    out(fmt_offers(ranked, 5, m=m, down_gb=down_gb))
    env = instance_env(job, project, creds, commit, ep)
    for rate, od, offer in ranked[:5]:
        # re-check the cap against the offer as rented, not as first listed
        if rate > m.max_price_per_gpu * m.num_gpus:
            raise RuntimeError(f"rate {rate} over cap — refusing")
        body = {"client_id": "me", "image": m.image, "disk": m.min_disk_gb, "onstart": onstart_script(),
                "env": env, "runtype": "ssh"}
        if not od:
            body["price"] = rate        # a price makes the contract interruptible
        try:
            inst = vast.rent(offer["id"], body)
        except VastError as ex:
            out(f"  rent of offer {offer['id']} failed ({ex}); trying the next")
            continue
        out(f"episode {ep}: rented {inst} at ${rate:.3f}/h ({'on-demand' if od else 'interruptible'}) "
            f"{offer.get('gpu_name')} {offer.get('geolocation')}")
        return inst, rate, od, offer
    raise RuntimeError("could not rent any of the top 5 offers")
