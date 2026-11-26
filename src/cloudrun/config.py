"""Project and job specs (TOML). Unknown keys are errors: a typo'd budget key must not silently mean 'no cap'."""

from __future__ import annotations

import dataclasses as dc
import tomllib
from pathlib import Path


def _build(cls, d: dict, where: str):
    names = {f.name for f in dc.fields(cls)}
    extra = set(d) - names
    if extra:
        raise ValueError(f"{where}: unknown keys {sorted(extra)} (allowed: {sorted(names)})")
    missing = [f.name for f in dc.fields(cls)
               if f.name not in d and f.default is dc.MISSING and f.default_factory is dc.MISSING]
    if missing:
        raise ValueError(f"{where}: missing required keys {missing}")
    return cls(**d)


@dc.dataclass(frozen=True)
class Project:
    bucket: str                 # B2 bucket; everything lives under <root>/runs/<job>/ and <root>/data/
    b2_env: str                 # local env file with B2_KEY_ID / B2_APPLICATION_KEY / B2_ENDPOINT
    vast_env: str               # local env file with VAST_API_KEY
    repo: str                   # public GitHub "owner/name"; instances fetch the commit tarball
    state_dir: str              # local supervisor state (spend, episodes) per job
    instance_b2_env: str = ""   # key shipped to instances; defaults to b2_env
    root: str = ""              # key prefix inside the bucket (lets projects share one bucket)

    def key(self, *parts: str) -> str:
        return "/".join(p.strip("/") for p in (self.root, *parts) if p and p.strip("/"))

    def path(self, s: str) -> Path:
        return Path(s).expanduser()


@dc.dataclass(frozen=True)
class Machine:
    gpus: list[str]                     # acceptable Vast gpu_name values
    max_price_per_gpu: float            # $/GPU-hour ceiling on what we would actually pay
    num_gpus: int = 1
    interruptible: bool = True
    bid_multiplier: float = 1.25        # bid = cheapest floor * this (capped): a floor bid is sniped at once
    min_reliability: float = 0.98
    min_disk_gb: int = 60
    min_gpu_ram_gb: float = 20
    min_cpu_cores: int = 8
    min_ram_gb: float = 32
    min_inet_down_mbps: float = 300
    min_cuda: float = 12.6              # host driver's max CUDA; the image needs at least this
    verified_only: bool = True
    image: str = "pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime"
    # offers are ranked by the cost of one episode, not the hourly rate: hosts charge per GB moved, and
    # pulling the job's data onto a fresh box can cost more than hours of GPU time
    episode_hours: float = 4.0          # expected GPU hours per episode
    upload_gb: float = 0.0              # expected GB synced to B2 per episode (checkpoints, logs)


@dc.dataclass(frozen=True)
class Data:
    remote: str     # prefix in the bucket (e.g. data/v2_128)
    local: str      # directory under $DATA on the instance


@dc.dataclass(frozen=True)
class Budget:
    max_spend: float            # HARD cap in $, across all episodes of the job
    max_hours: float = 24.0     # wall-clock cap for the supervisor


@dc.dataclass(frozen=True)
class Watch:
    boot_min: float = 12            # no heartbeat at all (image pull / boot hung) for this long -> replace
    setup_grace_min: float = 45     # no step yet this episode for this long -> replace
    stall_min: float = 20           # step stopped advancing for this long -> replace
    hb_stale_min: float = 12        # heartbeat content unchanged this long (syncer writes every minute) -> replace
    poll_s: float = 60
    max_failures: int = 2           # crashes/stalls after progress before giving up


@dc.dataclass(frozen=True)
class Job:
    name: str                       # B2 prefix runs/<name>/; also the supervisor state file
    command: str                    # shell; sees $RUN_DIR $DATA $WORK; must resume from $RUN_DIR itself
    progress_regex: str             # group 1 = step, matched against the job log
    machine: Machine
    budget: Budget
    setup: list[str] = dc.field(default_factory=list)
    data: list[Data] = dc.field(default_factory=list)
    watch: Watch = dc.field(default_factory=Watch)
    rolling: str = "ckpt_*.pt"      # run-dir files pruned remotely beyond the newest `keep_rolling`
    keep_rolling: int = 2
    env: dict = dc.field(default_factory=dict)


def load_project(path: str | Path) -> Project:
    p = Path(path)
    return _build(Project, tomllib.loads(p.read_text()), str(p))


def load_job(path: str | Path) -> Job:
    p = Path(path)
    d = tomllib.loads(p.read_text())
    where = str(p)
    if set(d) - {"job", "machine", "budget", "watch", "data"}:
        raise ValueError(f"{where}: unknown tables {sorted(set(d) - {'job', 'machine', 'budget', 'watch', 'data'})}")
    for t in ("job", "machine", "budget"):
        if t not in d:
            raise ValueError(f"{where}: missing [{t}]")
    job = dict(d["job"])
    job["machine"] = _build(Machine, d["machine"], f"{where} [machine]")
    job["budget"] = _build(Budget, d["budget"], f"{where} [budget]")
    job["watch"] = _build(Watch, d.get("watch", {}), f"{where} [watch]")
    job["data"] = [_build(Data, x, f"{where} [[data]]") for x in d.get("data", [])]
    j = _build(Job, job, f"{where} [job]")
    import re
    if re.compile(j.progress_regex).groups < 1:
        raise ValueError(f"{where}: progress_regex needs a capture group for the step")
    if not j.machine.gpus:
        raise ValueError(f"{where}: machine.gpus is empty")
    return j
