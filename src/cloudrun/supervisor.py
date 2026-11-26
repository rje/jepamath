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
  no heartbeat and no pulse, episode older than watch.boot_min -> replace (failure: image pull / boot hung)
  pulse seen, then pulse, heartbeat and step all unchanged
    for watch.hb_stale_min                                     -> replace (failure: box frozen or offline)
  not started, episode older than watch.setup_grace_min        -> replace (failure)
  failures > watch.max_failures                                -> stop
"""

from __future__ import annotations

import base64
import dataclasses as dc
import json
import os
import signal
import time
import uuid
from pathlib import Path

from . import b2
from .config import Job, Project
from .creds import B2Creds
from .vast import RUNNING_STATES, Vast, VastError, bandwidth_usd, fmt_offers, rank


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
    t_pulse: float | None = None  # local time the liveness pulse last changed (None: never seen; cloudrun.pulse)
    t_hb: float | None = None     # local time the syncer heartbeat's `time` last changed (None: never seen)
    pulse_unread: bool = False    # the pulse GET failed this poll: no boot-hang / wedge decision rests on it


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
    if o.t_pulse is not None and not o.pulse_unread:
        # any sign of life counts: the pulse, a fresh syncer heartbeat, or a new training step. Only when all of
        # them have been still for hb_stale_min is the box frozen or offline (a dead pulse process alone is not).
        signs = [t for t in (o.t_pulse, o.t_hb, o.t_step if o.ep_step > 0 else None) if t is not None]
        if (o.now - max(signs)) / 60 > w.hb_stale_min:
            phase = (o.hb or {}).get("phase", "no heartbeat")
            return "replace_failure", (f"pulse, heartbeat and step unchanged for {w.hb_stale_min} min (phase {phase}, "
                                       f"step {o.ep_step}): box wedged")
    if o.ep_step > 0:
        if (o.now - o.t_step) / 60 > w.stall_min:
            return "replace_failure", f"stalled: step {o.ep_step} unchanged for {w.stall_min} min"
    elif o.hb is None and o.t_pulse is None and not o.pulse_unread and (o.now - o.ep_start) / 60 > w.boot_min:
        return "replace_failure", f"no heartbeat {w.boot_min} min after renting (status {o.status}): boot hung"
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
# liveness pulse for the life of the box, independent of uploads (cloudrun.pulse; decide: hb_stale_min)
(cd $WORK/code/src && PYTHONPATH=$WORK/code/src nohup python -m cloudrun.pulse > $WORK/pulse.log 2>&1 &)
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


def pulse(s3, project: Project, job: str, ep: int) -> float | None:
    """The instance's liveness pulse time (cloudrun.pulse), None before the first one."""
    r = b2.get_bytes(s3, project.bucket, project.key("runs", job, "pulse", f"ep{ep:03d}.json"))
    return json.loads(r[0]).get("time") if r else None


class _Terminate(Exception):
    pass


class DestroyFailed(RuntimeError):
    pass


class NoOffers(RuntimeError):
    pass


class RentUnresolved(RuntimeError):
    """A rent may have reached Vast but cannot be found by its label yet: renting another box could leave
    two billing with one unrecorded, so the supervisor stops and st.d["renting"] is kept for the next start."""


RENT_SETTLE_S = 600          # how long the Vast instance list may lag a rent before an absent label means "not rented"
RENT_LOOKUP_TRIES, RENT_LOOKUP_WAIT_S = 4, 15


class job_lock:
    """One supervisor per job: an exclusive, non-blocking lock on <state_dir>/<job>.lock, held from before
    recovery until the supervisor exits. A second `launch --yes` of the same job fails instead of destroying
    the first supervisor's box or overwriting its ledger."""

    def __init__(self, state_dir: Path, job: str):
        self.path = Path(state_dir) / f"{job}.lock"

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w")
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.f.close()
            raise SystemExit(f"another supervisor holds {self.path}: is `launch {self.path.stem}` already running?")
        self.f.write(str(os.getpid()))
        self.f.flush()
        return self

    def __exit__(self, *_):
        self.f.close()
        return False


_GUARDED = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class _signals_deferred:
    """Block SIGINT/SIGTERM/SIGHUP for a bounded critical section (rent + ledger write, destroy + ledger
    write). A signal that arrives meanwhile is delivered on exit, so the section always completes."""

    def __enter__(self):
        self.old = signal.pthread_sigmask(signal.SIG_BLOCK, _GUARDED)

    def __exit__(self, *_):
        signal.pthread_sigmask(signal.SIG_SETMASK, self.old)
        return False


LATE_LABEL_DAYS = 7
_LEGACY_STOP = ("spend cap", "wall-clock cap", "job failed (rc=")


def _adopt_late(st: "State", vast: Vast, out, listing: list | None = None) -> list[dict]:
    """Rents that reached Vast but were never recorded, found by label: the pending rent (st.d["renting"]) and
    every label kept in st.d["late_labels"] (rents not listed when last checked; the Vast list can lag). Each
    one found is appended to the ledger as an open episode, which the caller destroys. Labels are kept for
    LATE_LABEL_DAYS, so a box that shows up late is still caught at the next launch, rent or exit."""
    labels = ([st.d["renting"]] if st.d.get("renting") else []) + list(st.d.get("late_labels", []))
    if not labels:
        return []
    listing = vast.instances() if listing is None else listing
    known = {e["instance"] for e in st.d["episodes"]}
    new = []
    for pend in labels:
        for i in listing:
            if i.get("label") == pend["label"] and i.get("id") not in known:
                out(f"rent {pend['label']} (offer {pend['offer']}) reached Vast as instance {i['id']} but was never "
                    f"recorded: adding it to the ledger and destroying it")
                dph = i.get("dph_total")
                e = {"n": len(st.d["episodes"]), "instance": i["id"],
                     "rate": max(pend["rate"], float(dph) if dph is not None else 0.0),
                     "bandwidth": bandwidth_usd(i), "on_demand": pend["od"],
                     "offer": pend["offer"], "machine": pend.get("machine"), "start": pend["time"], "end": None,
                     "end_reason": None, "max_step": 0, "commit": pend.get("commit"), "label": pend["label"]}
                st.d["episodes"].append(e)
                new.append(e)
                known.add(i["id"])
                pend["adopted"] = True
    if st.d.get("renting") and st.d["renting"].get("adopted"):
        st.d["renting"] = None          # reconciled: cleared in the same save that records the box
    st.d["late_labels"] = [p for p in st.d.get("late_labels", [])
                           if not p.get("adopted") and time.time() - p["time"] < LATE_LABEL_DAYS * 86400]
    st.save()
    return new


def recover(st: "State", vast: Vast, out) -> str | None:
    """Close every episode a previous supervisor left open: the box was rented but never confirmed destroyed
    (orphaned, or destroy_pending after failed destroys), or a rent whose reply was lost (st.d["renting"] and
    st.d["late_labels"], found by label). Runs before anything else at launch. If a saved decision says the job
    was done or stopped, that outcome is returned and nothing new is rented."""
    pend = st.d.get("renting")
    found = _adopt_late(st, vast, out)
    if pend and st.d.get("renting"):
        if not any(e["label"] == pend["label"] for e in found):
            if time.time() - pend["time"] < RENT_SETTLE_S:
                raise RentUnresolved(f"a rent ({pend['label']}, offer {pend['offer']}) from "
                                     f"{(time.time() - pend['time']) / 60:.0f} min ago is not listed yet; the Vast "
                                     f"list can lag, so wait until {RENT_SETTLE_S // 60} min have passed and launch again")
            # most likely never rented; the label is kept, so a late-listed box is still adopted and destroyed
            st.d.setdefault("late_labels", []).append(pend)
        st.d["renting"] = None
        st.save()
    outcome = None
    for e in st.d["episodes"]:
        # destroy_pending with `end` already set is the pre-c42068f format (end and spend were banked before the
        # failed destroy): still possibly billing, so it is destroyed too, and only the unbanked time is added
        if e["end"] is None or e.get("destroy_pending"):
            action = e.get("pending_action")
            if action is None and e.get("end") is not None:      # legacy: the decision is in end_reason
                r = e.get("end_reason") or ""
                action = "done" if r == "job exited 0" else "stop" if r.startswith(_LEGACY_STOP) else None
            reason = e.get("pending_reason") or e.get("end_reason") or "orphaned by a previous supervisor"
            out(f"episode {e['n']} (instance {e['instance']}) was left open ({reason}); destroying it")
            _end_episode(st, e, vast, e["instance"], reason, out, action)
            if action in ("done", "stop"):
                outcome = f"{action}: {reason}"
    if outcome:
        st.d["outcome"] = "done" if outcome.startswith("done") else outcome
        st.save()
    return outcome


def run(job: Job, project: Project, vast: Vast, s3, creds: B2Creds, commit: str, out=print) -> str:
    """Supervise until done/stop. Returns the outcome string. The current instance is destroyed on every exit
    path, including SIGTERM/SIGHUP (converted to an exception so `finally` runs); destroys and rents run with
    those signals deferred so a signal cannot cut one off between Vast and the ledger."""
    st = State(project.path(project.state_dir) / f"{job.name}.json", job.name)
    if st.d.get("outcome") == "done":
        raise RuntimeError(f"{job.name} is already done (state {st.path}); use a new job name to rerun")
    st.d["outcome"] = None
    prior = recover(st, vast, out)
    if prior:
        out(f"{job.name}: {prior} (decided before the previous supervisor's destroy failed)")
        return st.d["outcome"]
    any_progress = any(e.get("max_step", 0) > 0 for e in st.d["episodes"])

    def _sig(signum, _frame):
        raise _Terminate(f"signal {signum}")
    for s_ in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(s_, _sig)

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
                if (time.time() - t_start) / 3600 > job.budget.max_hours:
                    outcome = f"stop: wall-clock cap {job.budget.max_hours} h"
                    break
                ep = len(st.d["episodes"])
                try:
                    inst, e = provision(job, project, vast, creds, commit, ep, out, data_gb(job, project, s3),
                                        failed_hosts(project.path(project.state_dir), job.name, out=out), st)
                except NoOffers as ex:
                    out(f"  {ex}; looking again in {NO_OFFER_WAIT_S // 60} min")
                    time.sleep(NO_OFFER_WAIT_S)
                    continue
                except VastError as ex:     # search or late-label listing failed before any rent: retry later
                    out(f"  vast API error before renting ({ex}); trying again in {NO_OFFER_WAIT_S // 60} min")
                    time.sleep(NO_OFFER_WAIT_S)
                    continue
                ep = e["n"]          # provision may have adopted late rents first, shifting the number
                ep_step, t_step = 0, e["start"]
                pulse_last, t_pulse, hb_last, t_hb = None, None, None, None
                b2_down_since = None
            time.sleep(job.watch.poll_s)
            now = time.time()
            running = e["rate"] * (now - e["start"]) / 3600 + e.get("bandwidth", 0.0)
            cap = (f"spend cap: ${st.d['spent'] + running:.2f} >= ${job.budget.max_spend:.2f}"
                   if st.d["spent"] + running >= job.budget.max_spend else
                   f"wall-clock cap {job.budget.max_hours} h" if (now - t_start) / 3600 > job.budget.max_hours else None)
            try:
                status, billed, bw = vast.poll(inst)
            except VastError as ex:
                # an API hiccup is not evidence the box died; look again next poll. The caps still hold, judged
                # from local time and the last known rate, so an outage cannot bill forever.
                out(f"  vast API error ({ex})" + (f"; {cap}: tearing down" if cap else "; ignored this poll"))
                if not cap:
                    continue
                action, reason = "stop", cap
            else:
                if billed is not None and billed > e["rate"]:   # account at what Vast bills, never below the bid
                    e["rate"] = billed
                e["bandwidth"] = max(e.get("bandwidth", 0.0), bw)
                running = e["rate"] * (now - e["start"]) / 3600 + e.get("bandwidth", 0.0)   # with this poll's charges
                # the two B2 reads fail independently: a pulse read error never hides a readable heartbeat
                # (exit status, step); without the heartbeat no liveness rule runs, the caps still hold, and a
                # long blind spell stops the job (it cannot be supervised).
                hb, hb_err = _read(heartbeat, s3, project, job.name, ep)
                pt, pt_err = _read(pulse, s3, project, job.name, ep)
                for err in (hb_err, pt_err):
                    if err:
                        out(f"  B2 read error ({err})")
                if hb_err:
                    b2_down_since = b2_down_since or now
                    blind = (now - b2_down_since) / 60
                    spent = st.d["spent"] + running
                    cap = (f"spend cap: ${spent:.2f} >= ${job.budget.max_spend:.2f}" if spent >= job.budget.max_spend
                           else cap or (f"B2 heartbeat unreadable for {blind:.0f} min: cannot supervise"
                                        if blind > B2_OUTAGE_MIN else None))
                    if not cap:
                        out(f"  heartbeat unreadable for {blind:.0f} min; no liveness rule this poll")
                        continue
                    action, reason = "stop", cap
                else:
                    b2_down_since = None
                    if pt is not None and pt != pulse_last:
                        pulse_last, t_pulse = pt, now
                    if hb and hb.get("time") != hb_last:
                        hb_last, t_hb = hb.get("time"), now
                    if hb and hb.get("step", 0) > ep_step:
                        ep_step, t_step = hb["step"], now
                        e["max_step"] = ep_step
                        any_progress = True
                    # "unknown" = listed without an actual_status yet (vast.poll): alive, the boot rules cover it
                    o = Obs(now=now, t_start=t_start, ep_start=e["start"],
                            status="running" if status == "unknown" else status, hb=hb, ep_step=ep_step,
                            t_step=t_step, spent=st.d["spent"] + running, any_progress=any_progress,
                            failures=st.d["failures"], t_pulse=t_pulse, t_hb=t_hb,
                            pulse_unread=pt_err is not None)
                    action, reason = decide(o, job)
                    phase = (hb or {}).get("phase", "-")
                    out(f"  [ep{ep} {(now - e['start']) / 60:5.0f}m] {status or 'gone':<8} phase {phase:<8} "
                        f"step {ep_step:>7} ${o.spent:7.2f}/{job.budget.max_spend:.2f}"
                        + (f"  -> {action}: {reason}" if reason else ""))
            st.save()
            if action == "continue":
                continue
            inst = None              # the episode is in the ledger; the destroy below or `finally` closes it
            for round_ in range(DESTROY_ROUNDS):
                try:
                    _end_episode(st, e, vast, e["instance"], reason, out, action)
                    break
                except DestroyFailed:
                    if round_ + 1 == DESTROY_ROUNDS:
                        raise
                    out(f"  destroy round {round_ + 1}/{DESTROY_ROUNDS} failed; trying again")
            if action in ("done", "stop"):
                outcome = f"{action}: {reason}"
                break
    except (DestroyFailed, RentUnresolved) as ex:
        outcome = f"stopped: {ex}"
    except _Terminate as ex:
        outcome = f"stopped: {ex}"
    except KeyboardInterrupt:
        outcome = "stopped: interrupted"
    finally:
        # close EVERY episode the ledger still has open, not just the one in hand: a signal delivered as a rent's
        # deferral ends can leave a recorded box whose instance id never reached this frame; late-listed rents too
        try:
            _adopt_late(st, vast, out)
        except VastError as ex:
            out(f"  could not check late-listed rents ({ex}); the next start does")
        for e_ in [x for x in st.d["episodes"] if x["end"] is None or x.get("destroy_pending")]:
            try:
                _end_episode(st, e_, vast, e_["instance"], e_.get("pending_reason") or outcome, out,
                             e_.get("pending_action"))
            except DestroyFailed as ex:
                outcome = f"stopped: {ex}"
        still = [x["instance"] for x in st.d["episodes"] if x["end"] is None or x.get("destroy_pending")]
        if still:
            out(f"WARNING: instance(s) {still} could not be confirmed destroyed and may be billing; the next "
                f"`launch --yes` destroys them before anything else")
        if st.d.get("outcome") is None:      # a done/stop already saved with its episode's closure stands
            st.d["outcome"] = "done" if outcome.startswith("done") else outcome
        st.save()
        outcome = st.d["outcome"]           # the summary says what the ledger says
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


DESTROY_TRIES, DESTROY_WAIT_S = 6, 30
DESTROY_ROUNDS = 2           # rounds of DESTROY_TRIES inside the loop before the supervisor gives up and stops
NO_OFFER_WAIT_S = 300
B2_OUTAGE_MIN = 30


def _read(fn, *args):
    """(value, None) or (None, "Type: message") for a B2 read; only B2/network errors are caught."""
    from botocore.exceptions import BotoCoreError, ClientError     # here, not at import: recovery never needs it
    try:
        return fn(*args), None
    except (BotoCoreError, ClientError, OSError) as ex:
        return None, f"{type(ex).__name__}: {ex}"


def _end_episode(st: State, e: dict, vast: Vast, inst: int, reason: str, out, action: str | None = None) -> None:
    """Destroy the box, then close the episode in the ledger, with signals deferred throughout.

    The decision (`action`, `reason`) is saved with destroy_pending BEFORE the first DELETE. On success the
    episode's end is set now, so its spend runs to the verified destroy, and a replace_failure counts once. If
    all DESTROY_TRIES fail, the episode stays open (end None: still billing) and DestroyFailed is raised; the next
    supervisor start (recover) retries it and applies the saved decision."""
    with _signals_deferred():
        legacy = e["end"] is not None
        if legacy:
            # pre-c42068f record: spend banked to e["end"]. If the box is already gone we cannot know when it
            # went, so nothing is added (check the Vast billing page); if it is still listed it bills until now.
            try:
                absent = vast.status(inst) is None
            except VastError:
                absent = False
        e["destroy_pending"], e["pending_action"], e["pending_reason"] = True, action, reason
        st.save()
        out(f"  destroying {inst} ({reason})")
        err = None
        for i in range(DESTROY_TRIES):
            try:
                vast.destroy(inst)
                out("  destroyed (verified)")
                err = None
                break
            except VastError as ex:
                err = ex
                out(f"  destroy attempt {i + 1}/{DESTROY_TRIES} failed: {ex}")
                if i + 1 < DESTROY_TRIES:
                    time.sleep(DESTROY_WAIT_S)
        if err is not None:
            st.save()
            raise DestroyFailed(f"instance {inst} could not be destroyed after {DESTROY_TRIES} tries ({err}): it may "
                                f"still be billing. Kill it at https://cloud.vast.ai/instances/ ; the next "
                                f"supervisor start retries it")
        now = time.time()
        if not legacy:
            st.d["spent"] += e["rate"] * (now - e["start"]) / 3600 + e.get("bandwidth", 0.0)
            e["end_reason"] = reason
        elif absent:
            e["end_reason"] = f"{e.get('end_reason')}; already gone at recovery (gap after end not billed here)"
            e["unbanked_gap"] = True
            out(f"  WARNING: instance {inst} was already gone; any billing between its recorded end and its "
                f"removal is NOT in the ledger. Check the Vast billing page.")
        else:       # pre-c42068f destroy_pending, still listed: add only the time since the banked end
            st.d["spent"] += e["rate"] * (now - e["end"]) / 3600
            e["end_reason"] = f"{e.get('end_reason')}; destroy confirmed later"
        e["end"] = now
        if action == "replace_failure":
            st.d["failures"] += 1
        if action in ("done", "stop"):      # the job's outcome is saved in the same write as the closure
            st.d["outcome"] = "done" if action == "done" else f"stop: {reason}"
        e["destroy_pending"] = False
        st.save()


FAILED_HOST_HOURS = 24.0


def failed_hosts(state_dir: Path, job: str, now: float | None = None, hours: float = FAILED_HOST_HOURS,
                 out=print) -> set:
    """Hosts not to rent, as ("offer", id) / ("machine", id), from episodes that ended within `hours`:
    - boot hung (no sign of life at all: image pull or the box itself), from ANY job's state;
    - wedged (pulse stopped) or no progress within the setup grace, from THIS job's state only, since a slow
      dataset or a B2 outage can cause those on a healthy host.
    Preemptions never count (the market, not the host). An unreadable peer state file is skipped with a
    warning; this job's own must parse. Renting the same host straight after a hang was 11 of 43 boot hangs."""
    now = time.time() if now is None else now
    bad = set()
    for f in Path(state_dir).glob("*.json"):
        own = f.stem == job
        try:
            d = json.loads(f.read_text())
            if not isinstance(d, dict) or not isinstance(d.get("episodes", []), list) \
                    or not all(isinstance(e, dict) for e in d.get("episodes", [])):
                raise ValueError("not a cloudrun state (expected an object with an episodes list)")
            eps = d.get("episodes", [])
        except (json.JSONDecodeError, OSError, ValueError) as ex:
            if own:
                raise
            out(f"  warning: skipping unreadable state {f.name} ({ex})")
            continue
        for e in eps:
            r = e.get("end_reason") or ""
            if e.get("end") is None:
                continue
            if not isinstance(e["end"], (int, float)) or not isinstance(r, str):
                if own:
                    raise ValueError(f"{f.name}: episode {e.get('n')} has end={e['end']!r} / end_reason={r!r}")
                out(f"  warning: skipping malformed episode {e.get('n')} in {f.name}")
                continue
            if now - e["end"] > hours * 3600:
                continue
            if "boot hung" in r or (own and ("wedged" in r or "no progress" in r)):
                for kind in ("offer", "machine"):
                    v = e.get(kind)
                    if isinstance(v, int) and not isinstance(v, bool):
                        bad.add((kind, v))
                    elif v is not None:
                        out(f"  warning: ignoring malformed {kind} id {v!r} in {f.name}")
    return bad


def data_gb(job, project, s3) -> float:
    """GB a fresh box pulls: the job's data plus its run dir (restore)."""
    prefixes = [project.key(d.remote) + "/" for d in job.data] + [project.key("runs", job.name, "run") + "/"]
    return sum(sum(b2.remote_sizes(s3, project.bucket, p).values()) for p in prefixes) / 1e9


def provision(job: Job, project: Project, vast: Vast, creds: B2Creds, commit: str, ep: int, out,
              down_gb: float = 0.0, avoid: set = frozenset(), st: State | None = None):
    """Rent the cheapest eligible offer not on a failed host. With a ledger (`st`), the rent runs with
    signals deferred and is recorded in st.d["renting"] (label, offer) BEFORE the request, then as an episode
    after it; a rent whose reply is lost is looked up by its label before trying the next offer. Returns
    (instance, episode dict) with a ledger, else (instance, rate, on_demand, offer)."""
    m = job.machine
    ranked = rank(vast.search_all(m), m, down_gb)
    n0 = len(ranked)
    ranked = [t for t in ranked if ("offer", t[2]["id"]) not in avoid and ("machine", t[2].get("machine_id")) not in avoid]
    if n0 != len(ranked):
        out(f"  skipping {n0 - len(ranked)} offer(s) on hosts that failed in the last {FAILED_HOST_HOURS:.0f} h")
    if n0 and not ranked:
        raise NoOffers(f"all {n0} eligible offers are on hosts that failed in the last {FAILED_HOST_HOURS:.0f} h")
    if not ranked:
        raise NoOffers(f"no offer for {m.gpus} at <= ${m.max_price_per_gpu}/GPU-h passes the filters")
    if st is not None:      # a rent that showed up late in the list is destroyed before another is made
        for late in _adopt_late(st, vast, out):
            _end_episode(st, late, vast, late["instance"], "late-listed rent adopted", out)
        # episode numbers are never reused, so a box whose rent was lost cannot share heartbeat keys with a new one
        ep = max(len(st.d["episodes"]), st.d.get("next_ep", 0))
    out(fmt_offers(ranked, 5, m=m, down_gb=down_gb))
    env = instance_env(job, project, creds, commit, ep)
    for rate, od, offer in ranked[:5]:
        # re-check the cap against the offer as rented, not as first listed
        if rate > m.max_price_per_gpu * m.num_gpus:
            raise RuntimeError(f"rate {rate} over cap — refusing")
        label = f"cloudrun:{job.name}:ep{ep}:{uuid.uuid4().hex[:8]}"
        body = {"client_id": "me", "image": m.image, "disk": m.min_disk_gb, "onstart": onstart_script(),
                "env": env, "runtype": "ssh", "label": label}
        if not od:
            body["price"] = rate        # a price makes the contract interruptible
        with _signals_deferred():
            if st is not None:
                st.d["renting"] = {"label": label, "offer": offer["id"], "machine": offer.get("machine_id"),
                                   "rate": rate, "od": od, "time": time.time(), "commit": commit, "ep": ep}
                st.d["next_ep"] = ep + 1
                st.save()
            try:
                inst = vast.rent(offer["id"], body)
            except VastError as ex:
                inst = None
                # A 4xx is a definite refusal (offer taken, bad bid). Anything else (network error, timeout, 5xx,
                # odd reply) may have rented a box, and the Vast list can lag a rent, so one empty listing is
                # not proof. Look it up by label a few times; still absent -> stop rather than rent a second box.
                definite = ex.definite
                if st is not None:
                    for k in range(0 if definite else RENT_LOOKUP_TRIES):
                        try:
                            hit = [i["id"] for i in vast.instances() if i.get("label") == label]
                        except VastError as ex2:
                            raise RentUnresolved(f"rent of offer {offer['id']} failed ({ex}) and the instance list "
                                                 f"is unreachable ({ex2}); the next start reconciles label {label}")
                        if hit:
                            inst = hit[0]
                            break
                        if not definite and k + 1 < RENT_LOOKUP_TRIES:
                            time.sleep(RENT_LOOKUP_WAIT_S)
                    if inst is None and not definite:
                        raise RentUnresolved(f"rent of offer {offer['id']} failed ambiguously ({ex}) and label "
                                             f"{label} is not listed after {RENT_LOOKUP_TRIES} looks; stopping so "
                                             f"the next start (after {RENT_SETTLE_S // 60} min) reconciles it")
                if inst is None:
                    out(f"  rent of offer {offer['id']} refused ({ex}); trying the next")
                    if st is not None:
                        st.d["renting"] = None
                        st.save()
                    continue
                out(f"  rent reply lost but instance {inst} exists with label {label}: adopting it")
            out(f"episode {ep}: rented {inst} at ${rate:.3f}/h ({'on-demand' if od else 'interruptible'}) "
                f"{offer.get('gpu_name')} {offer.get('geolocation')}")
            if st is None:
                return inst, rate, od, offer
            e = {"n": ep, "instance": inst, "rate": rate, "on_demand": od, "offer": offer["id"],
                 "machine": offer.get("machine_id"), "label": label, "gpu": offer.get("gpu_name"),
                 "geo": offer.get("geolocation"), "start": st.d["renting"]["time"], "end": None, "end_reason": None,
                 "max_step": 0, "commit": commit}
            st.d["episodes"].append(e)
            st.d["renting"] = None
            st.save()
            return inst, e
    raise NoOffers("could not rent any of the top 5 offers")
