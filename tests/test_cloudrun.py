"""cloudrun: every rule that decides whether money is spent, without renting or touching B2."""

from __future__ import annotations

import base64
import dataclasses as dc
import io
import json
import os
import re
import time
from pathlib import Path

import pytest

from cloudrun import supervisor as sup
from cloudrun.config import Budget, Job, Machine, Project, Watch, load_job
from cloudrun.creds import B2Creds, read_env_file
from cloudrun.sync import Syncer, parse_log, rolling_to_prune
from cloudrun.vast import bandwidth_usd, eligible, episode_cost, planned_rate, rank

ROOT = Path(__file__).resolve().parents[1]


class FakeS3:
    def __init__(self):
        self.obj: dict[str, tuple[bytes, float]] = {}

    def list_objects_v2(self, Bucket, Prefix="", **_):
        return {"Contents": [{"Key": k, "Size": len(v[0])} for k, v in sorted(self.obj.items()) if k.startswith(Prefix)],
                "IsTruncated": False}

    def put_object(self, Bucket, Key, Body):
        self.obj[Key] = (Body if isinstance(Body, bytes) else Body.encode(), time.time())

    def upload_file(self, path, Bucket, Key, **_):
        self.obj[Key] = (Path(path).read_bytes(), time.time())

    def delete_object(self, Bucket, Key):
        self.obj.pop(Key, None)

    def get_object(self, Bucket, Key):
        from botocore.exceptions import ClientError
        if Key not in self.obj:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        import datetime
        b, t = self.obj[Key]
        return {"Body": io.BytesIO(b), "LastModified": datetime.datetime.fromtimestamp(t)}


M = Machine(gpus=["RTX 4090"], max_price_per_gpu=0.40)


def offer(**kw):
    o = {"id": 1, "gpu_name": "RTX 4090", "num_gpus": 1, "reliability2": 0.99, "disk_space": 200,
         "gpu_ram": 24564, "cpu_cores_effective": 32, "cpu_ram": 128000, "inet_down": 900, "cuda_max_good": 12.8,
         "dph_total": 0.45, "min_bid": 0.20, "geolocation": "US"}
    o.update(kw)
    return o


def job(**kw):
    base = dict(name="t", command="true", progress_regex=r'"step": (\d+)', machine=M,
                budget=Budget(max_spend=5.0, max_hours=10), watch=Watch(setup_grace_min=40, stall_min=20))
    base.update(kw)
    return Job(**base)


# ---------- config ----------

def test_real_job_specs_load_and_resume():
    for f in sorted((ROOT / "cloud" / "jobs").glob("*.toml")):
        j = load_job(f)
        assert "--auto-resume" in j.command, f"{f.name}: a preemptible job must resume from $RUN_DIR"
        # 12: the 256 px object-label run needed $11 after outbid churn (23 Nov); anything larger is likely a typo
        assert j.budget.max_spend <= 12, f"{f.name}: test-run budget unexpectedly large"


def test_unknown_budget_key_is_an_error(tmp_path):
    p = tmp_path / "j.toml"
    p.write_text('[job]\nname="x"\ncommand="true"\nprogress_regex="(\\\\d+)"\n[machine]\ngpus=["RTX 4090"]\n'
                 'max_price_per_gpu=0.3\n[budget]\nmax_spendd=5\n')
    with pytest.raises(ValueError, match="unknown keys"):
        load_job(p)


def test_progress_regex_must_capture(tmp_path):
    p = tmp_path / "j.toml"
    p.write_text('[job]\nname="x"\ncommand="true"\nprogress_regex="step"\n[machine]\ngpus=["RTX 4090"]\n'
                 'max_price_per_gpu=0.3\n[budget]\nmax_spend=5\n')
    with pytest.raises(ValueError, match="capture group"):
        load_job(p)


def test_progress_regex_matches_the_trainer_log():
    j = load_job(ROOT / "cloud" / "jobs" / "trackb-smoke.toml")
    line = '[train] {"step": 500, "lr": 0.0004, "clips_per_s": 92.3}'
    assert parse_log(line, re.compile(j.progress_regex))["step"] == 500


def test_credential_file_must_be_private(tmp_path):
    p = tmp_path / "x.env"
    p.write_text("VAST_API_KEY=abc\n")
    os.chmod(p, 0o644)
    with pytest.raises(PermissionError):
        read_env_file(p)
    os.chmod(p, 0o600)
    assert read_env_file(p)["VAST_API_KEY"] == "abc"


def test_creds_repr_hides_the_secret():
    c = B2Creds("0011223344", "SECRETSECRET", "https://s3.example")
    assert "SECRET" not in repr(c) and "SECRET" not in str(c)


# ---------- offers ----------

def test_bid_is_floor_times_multiplier():
    assert planned_rate(offer(), M) == (0.25, False)


def test_bid_is_capped_at_the_ceiling():
    assert planned_rate(offer(min_bid=0.39, dph_total=0.9), M) == (0.40, False)


def test_on_demand_wins_within_ten_percent():
    assert planned_rate(offer(min_bid=0.20, dph_total=0.27), M) == (0.27, True)


def test_nothing_over_the_cap():
    assert planned_rate(offer(min_bid=0.5, dph_total=0.9), M) is None


def test_on_demand_only_machine_caps_dph_total():
    od = dc.replace(M, interruptible=False)
    assert planned_rate(offer(dph_total=0.35), od) == (0.35, True)
    assert planned_rate(offer(dph_total=0.45), od) is None


@pytest.mark.parametrize("bad", [dict(reliability2=0.9), dict(cpu_cores_effective=4), dict(cpu_ram=16000),
                                 dict(inet_down=100), dict(cuda_max_good=12.2), dict(gpu_ram=12000),
                                 dict(gpu_name="RTX 3060"), dict(num_gpus=2), dict(disk_space=20)])
def test_hardware_filters(bad):
    assert eligible([offer(**bad)], M) == []


def test_rank_is_cheapest_planned_rate_first():
    r = rank([offer(id=1, min_bid=0.30, dph_total=0.9), offer(id=2, min_bid=0.10, dph_total=0.9)], M)
    assert [o["id"] for _, _, o in r] == [2, 1]


# ---------- decisions ----------

def obs(**kw):
    now = 10_000.0
    d = dict(now=now, t_start=now - 600, ep_start=now - 600, status="running", hb=None, ep_step=0, t_step=now - 600,
             spent=1.0, any_progress=False, failures=0)
    d.update(kw)
    return sup.Obs(**d)


def test_continue_while_healthy():
    assert sup.decide(obs(hb={"phase": "run", "step": 100, "exit_rc": None}, ep_step=100, t_step=9_990), job())[0] == "continue"


def test_spend_cap_counts_the_running_episode():
    assert sup.decide(obs(spent=5.0), job())[0] == "stop"


def test_wall_clock_cap():
    assert sup.decide(obs(t_start=10_000 - 11 * 3600), job())[0] == "stop"


def test_exit_zero_is_done():
    assert sup.decide(obs(hb={"exit_rc": 0}), job())[0] == "done"


def test_failure_before_any_progress_stops_rather_than_looping():
    assert sup.decide(obs(hb={"exit_rc": 1, "phase": "run"}), job())[0] == "stop"


def test_failure_after_progress_replaces():
    assert sup.decide(obs(hb={"exit_rc": 1, "phase": "run"}, any_progress=True), job())[0] == "replace_failure"


def test_exit_status_wins_over_a_stopped_container():
    # the box exits right after a clean finish; that is done, not a preemption
    assert sup.decide(obs(hb={"exit_rc": 0}, status="exited"), job())[0] == "done"


def test_exited_or_missing_instance_is_a_preemption():
    assert sup.decide(obs(status="exited"), job())[0] == "replace"
    assert sup.decide(obs(status=None), job())[0] == "replace"


def test_stall_after_progress():
    o = obs(ep_step=500, t_step=10_000 - 21 * 60, hb={"step": 500, "exit_rc": None})
    assert sup.decide(o, job())[0] == "replace_failure"


def test_setup_gets_its_own_longer_clock():
    assert sup.decide(obs(ep_start=10_000 - 30 * 60, hb={"phase": "setup", "step": 0}), job())[0] == "continue"     # 30 min: past stall, inside grace
    assert sup.decide(obs(ep_start=10_000 - 41 * 60, hb={"phase": "setup", "step": 0}), job())[0] == "replace_failure"


# ---------- instance env / onstart ----------

def test_instance_env_round_trips_multiline_values():
    j = job(command='python x.py --re "a b"', setup=["pip install a", "pip install b"])
    p = Project(bucket="b", b2_env="", vast_env="", repo="o/r", state_dir="/tmp")
    env = sup.instance_env(j, p, B2Creds("k", "s", "e"), "abc123", 3)
    assert base64.b64decode(env["CLOUDRUN_SETUP_B64"]).decode() == "pip install a\npip install b"
    assert base64.b64decode(env["CLOUDRUN_COMMAND_B64"]).decode() == j.command
    assert env["CLOUDRUN_EPISODE"] == "3" and env["CLOUDRUN_COMMIT"] == "abc123"
    assert all("\n" not in v for v in env.values())


def test_job_env_cannot_override_the_contract():
    j = job(env={"CLOUDRUN_BUCKET": "other"})
    p = Project(bucket="b", b2_env="", vast_env="", repo="o/r", state_dir="/tmp")
    with pytest.raises(ValueError):
        sup.instance_env(j, p, B2Creds("k", "s", "e"), "c", 0)


def test_onstart_hands_over_to_bootstrap_and_reports_fetch_failure():
    s = sup.onstart_script()
    assert "exec bash $WORK/code/src/cloudrun/bootstrap.sh" in s
    assert "CLOUDRUN_EXIT rc=94" in s and '"heartbeat", f"ep{ep:03d}.json"' in s
    assert (ROOT / "src" / "cloudrun" / "bootstrap.sh").exists()


def test_bootstrap_emits_the_exit_sentinel_on_every_path():
    s = (ROOT / "src" / "cloudrun" / "bootstrap.sh").read_text()
    for rc in ("finish 90", "finish 91", "finish 92", 'finish "$RC"'):
        assert rc in s
    assert 'echo "CLOUDRUN_EXIT rc=$rc"' in s


# ---------- syncer ----------

def test_parse_log_takes_the_last_phase_highest_step_and_exit():
    t = "CLOUDRUN_PHASE setup\nCLOUDRUN_PHASE run\n\"step\": 50\n\"step\": 100\nCLOUDRUN_EXIT rc=3\n"
    assert parse_log(t, re.compile(r'"step": (\d+)')) == {"phase": "run", "step": 100, "exit_rc": 3}
    assert parse_log("", re.compile(r'(\d+)'))["exit_rc"] is None


def test_rolling_prune_keeps_the_newest():
    keys = [f"runs/j/run/ckpt_{s:06d}.pt" for s in (500, 1000, 1500)] + ["runs/j/run/weights_002500.pt"]
    assert rolling_to_prune(keys, "ckpt_*.pt", 2) == ["runs/j/run/ckpt_000500.pt"]
    assert rolling_to_prune(keys, "ckpt_*.pt", 0) == []


def _syncer(tmp_path, s3, ep=0):
    run, log = tmp_path / "run", tmp_path / "job.log"
    run.mkdir(exist_ok=True)
    return Syncer(s3, "b", "j", ep, run, log, r'"step": (\d+)', "ckpt_*.pt", 2), run, log


def test_syncer_ships_stable_files_and_per_episode_heartbeats(tmp_path, monkeypatch):
    s3 = FakeS3()
    s, run, log = _syncer(tmp_path, s3, ep=4)
    (run / "log.jsonl").write_text("x")
    old = time.time() - 60
    os.utime(run / "log.jsonl", (old, old))
    (run / "ckpt_000100.pt").write_text("fresh")            # just written: not stable yet
    log.write_text('CLOUDRUN_PHASE run\n"step": 100\n')
    hb = s.once()
    assert "runs/j/run/log.jsonl" in s3.obj and "runs/j/run/ckpt_000100.pt" not in s3.obj
    assert hb["step"] == 100 and hb["episode"] == 4
    assert json.loads(s3.obj["runs/j/heartbeat/ep004.json"][0])["phase"] == "run"
    assert "runs/j/logs/ep004.log" in s3.obj


def test_syncer_reuploads_a_changed_file_and_prunes_rolling(tmp_path):
    s3 = FakeS3()
    s, run, log = _syncer(tmp_path, s3)
    old = time.time() - 60
    for st in (100, 200, 300):
        p = run / f"ckpt_{st:06d}.pt"
        p.write_text(str(st))
        os.utime(p, (old, old))
    s.once()
    assert sorted(k for k in s3.obj if "ckpt_" in k) == ["runs/j/run/ckpt_000200.pt", "runs/j/run/ckpt_000300.pt"]
    f = run / "log.jsonl"
    f.write_text("a"); os.utime(f, (old, old)); s.once()
    f.write_text("ab"); os.utime(f, (old + 1, old + 1)); s.once()
    assert s3.obj["runs/j/run/log.jsonl"][0] == b"ab"


def test_heartbeat_step_never_goes_backwards(tmp_path):
    s3 = FakeS3()
    s, run, log = _syncer(tmp_path, s3)
    log.write_text('"step": 500\n')
    assert s.once()["step"] == 500
    log.write_text("tail lost the old lines\n")
    assert s.once()["step"] == 500


# ---------- the loop, end to end with fakes ----------

class FakeVast:
    def __init__(self, script):
        self.script = script        # per instance: list of statuses returned per poll
        self.live: dict[int, list[str]] = {}
        self.rented, self.destroyed, self.next_id = [], [], 100

    def search_all(self, m):
        return [offer(id=7)]

    def rent(self, offer_id, body):
        self.next_id += 1
        self.rented.append((self.next_id, body))
        self.live[self.next_id] = list(self.script.pop(0))
        return self.next_id

    def status(self, inst):
        q = self.live.get(inst)
        if q is None:
            return None
        return q.pop(0) if len(q) > 1 else q[0]

    def poll(self, inst):
        return self.status(inst), self.billed, self.bw

    billed, bw = None, 0.0

    def instances(self):
        return [{"id": i, "actual_status": "running"} for i in self.live]

    def destroy(self, inst):
        self.destroyed.append(inst)
        self.live.pop(inst, None)


def test_loop_replaces_a_preempted_box_then_finishes(tmp_path, monkeypatch):
    s3 = FakeS3()
    vast = FakeVast([["running", "exited"], ["running", "running"]])
    j = job(watch=Watch(poll_s=0, setup_grace_min=40, stall_min=20))
    p = Project(bucket="b", b2_env="", vast_env="", repo="o/r", state_dir=str(tmp_path))
    polls = {"n": 0}
    real_hb = sup.heartbeat

    def hb(s3_, project, name, ep):
        polls["n"] += 1
        if ep == 1 and polls["n"] >= 4:
            s3.put_object("b", f"runs/{name}/heartbeat/ep001.json",
                          json.dumps({"episode": 1, "step": 600, "exit_rc": 0, "phase": "done"}))
        return real_hb(s3_, project, name, ep)
    monkeypatch.setattr(sup, "heartbeat", hb)
    out = sup.run(j, p, vast, s3, B2Creds("k", "s", "e"), "c0ffee", out=lambda m: None)
    assert out == "done"
    assert len(vast.rented) == 2 and set(vast.destroyed) == {101, 102}
    st = json.loads((tmp_path / "t.json").read_text())
    assert [e["end_reason"] for e in st["episodes"]][0].startswith("instance exited")
    assert st["failures"] == 0 and all(e["end"] for e in st["episodes"])
    assert "price" in vast.rented[0][1] and vast.rented[0][1]["price"] == 0.25    # interruptible bid


def test_loop_destroys_the_box_when_interrupted(tmp_path, monkeypatch):
    s3 = FakeS3()
    vast = FakeVast([["running"]])
    j = job(watch=Watch(poll_s=0))
    p = Project(bucket="b", b2_env="", vast_env="", repo="o/r", state_dir=str(tmp_path))

    def boom(*_a, **_k):
        raise KeyboardInterrupt
    monkeypatch.setattr(sup, "heartbeat", boom)
    out = sup.run(j, p, vast, s3, B2Creds("k", "s", "e"), "c", out=lambda m: None)
    assert out.startswith("stopped") and vast.destroyed == [101]


def test_an_orphaned_episode_is_destroyed_on_restart(tmp_path, monkeypatch):
    s3 = FakeS3()
    vast = FakeVast([["running"]])
    vast.live[55] = ["running"]
    (tmp_path / "t.json").write_text(json.dumps({"job": "t", "spent": 0.0, "failures": 0, "outcome": None,
        "episodes": [{"n": 0, "instance": 55, "rate": 0.3, "start": time.time() - 3600, "end": None,
                      "end_reason": None, "max_step": 0}]}))
    j = job(budget=Budget(max_spend=0.2, max_hours=1), watch=Watch(poll_s=0))
    p = Project(bucket="b", b2_env="", vast_env="", repo="o/r", state_dir=str(tmp_path))
    out = sup.run(j, p, vast, s3, B2Creds("k", "s", "e"), "c", out=lambda m: None)
    assert 55 in vast.destroyed and out.startswith("stop: spend cap") and vast.rented == []


# ---------- trainer checkpoints ----------

def test_trainer_checkpoints_are_atomic_and_pruned(tmp_path):
    from jepatrain.train import latest_ckpt, write_ckpt
    assert latest_ckpt(str(tmp_path)) is None
    for s in (100, 200, 300):
        write_ckpt(str(tmp_path), s, {"step": s}, keep=2)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ckpt_000200.pt", "ckpt_000300.pt"]
    assert latest_ckpt(str(tmp_path)).endswith("ckpt_000300.pt")


def test_root_prefixes_every_key(tmp_path):
    p = Project(bucket="b", b2_env="", vast_env="", repo="o/r", state_dir="/tmp", root="jepamath")
    assert p.key("runs", "j", "run") == "jepamath/runs/j/run"
    assert Project(bucket="b", b2_env="", vast_env="", repo="o/r", state_dir="/tmp").key("data/x/") == "data/x"
    s3 = FakeS3()
    run, log = tmp_path / "run", tmp_path / "job.log"
    run.mkdir()
    Syncer(s3, "b", "j", 2, run, log, r"(\d+)", "ckpt_*.pt", 2, root="jepamath").once()
    assert set(s3.obj) == {"jepamath/runs/j/logs/ep002.log", "jepamath/runs/j/heartbeat/ep002.json"}
    env = sup.instance_env(job(data=[__import__("cloudrun.config").config.Data("data/v2_128", "v2_128")]), p,
                           B2Creds("k", "s", "e"), "c", 0)
    assert base64.b64decode(env["CLOUDRUN_DATA_B64"]).decode() == "jepamath/data/v2_128 v2_128"
    assert env["CLOUDRUN_ROOT"] == "jepamath"


def test_final_flush_ships_a_file_written_a_moment_before_exit(tmp_path):
    s3 = FakeS3()
    s, run, log = _syncer(tmp_path, s3)
    (run / "result.json").write_text("{}")
    s.once()
    assert "runs/j/run/result.json" not in s3.obj
    s.once(final=True)
    assert "runs/j/run/result.json" in s3.obj


def test_spend_follows_the_billed_rate_not_the_bid(tmp_path, monkeypatch):
    s3 = FakeS3()
    vast = FakeVast([["running"]])
    vast.billed = 0.40                     # bid 0.25 + disk/bandwidth
    j = job(watch=Watch(poll_s=0))
    p = Project(bucket="b", b2_env="", vast_env="", repo="o/r", state_dir=str(tmp_path))
    s3.put_object("b", f"runs/{j.name}/heartbeat/ep000.json",
                  json.dumps({"episode": 0, "step": 600, "exit_rc": 0, "phase": "done"}))
    assert sup.run(j, p, vast, s3, B2Creds("k", "s", "e"), "c", out=lambda m: None) == "done"
    st = json.loads((tmp_path / "t.json").read_text())
    assert st["episodes"][0]["rate"] == 0.40


def test_rank_counts_the_data_pull_not_just_the_hourly_rate():
    m = dc.replace(M, episode_hours=4.0, upload_gb=0.0)
    cheap_gpu_dear_net = offer(id=1, min_bid=0.10, dph_total=0.9, inet_down_cost=0.05)   # 0.5 + 1.40
    dearer_gpu_free_net = offer(id=2, min_bid=0.20, dph_total=0.9, inet_down_cost=0.0)   # 1.0
    assert [o["id"] for _, _, o in rank([cheap_gpu_dear_net, dearer_gpu_free_net], m, down_gb=28)] == [2, 1]
    assert [o["id"] for _, _, o in rank([cheap_gpu_dear_net, dearer_gpu_free_net], m, down_gb=0)] == [1, 2]
    assert episode_cost(0.125, cheap_gpu_dear_net, m, 28) == 0.5 + 28 * 0.05


def test_bandwidth_usd_from_instance_record():
    i = {"inet_down_billed": 27682406.4, "inet_down_cost": 0.036458, "inet_up_billed": 5620367.36, "inet_up_cost": 0.03255}
    assert abs(bandwidth_usd(i) - (27.68 * 0.036458 + 5.62 * 0.03255)) < 0.01
    assert bandwidth_usd({}) == 0.0


def test_spend_includes_bandwidth(tmp_path):
    s3 = FakeS3()
    vast = FakeVast([["running"]])
    vast.bw = 1.25
    j = job(watch=Watch(poll_s=0))
    p = Project(bucket="b", b2_env="", vast_env="", repo="o/r", state_dir=str(tmp_path))
    s3.put_object("b", f"runs/{j.name}/heartbeat/ep000.json",
                  json.dumps({"episode": 0, "step": 600, "exit_rc": 0, "phase": "done"}))
    assert sup.run(j, p, vast, s3, B2Creds("k", "s", "e"), "c", out=lambda m: None) == "done"
    st = json.loads((tmp_path / "t.json").read_text())
    assert st["episodes"][0]["bandwidth"] == 1.25 and st["spent"] >= 1.25


def test_a_box_that_never_boots_is_replaced_after_boot_min():
    j = job(watch=Watch(boot_min=12, setup_grace_min=40))
    base = dict(now=0.0, t_start=0.0, ep_start=0.0, status="loading", hb=None, ep_step=0, t_step=0.0,
                spent=0.0, any_progress=False, failures=0)
    assert sup.decide(sup.Obs(**{**base, "now": 11 * 60}), j)[0] == "continue"
    assert sup.decide(sup.Obs(**{**base, "now": 13 * 60}), j)[0] == "replace_failure"
    booted = {**base, "now": 13 * 60, "hb": {"phase": "data", "step": 0}}
    assert sup.decide(sup.Obs(**booted), j)[0] == "continue"          # booted: setup_grace applies


def test_vast_network_error_is_vast_error(monkeypatch):
    """A DNS/timeout failure must surface as VastError, which the supervisor's poll loop retries."""
    import urllib.error
    import urllib.request
    import pytest
    from cloudrun.vast import Vast, VastError

    def boom(*a, **k):
        raise urllib.error.URLError("[Errno -3] Temporary failure in name resolution")
    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(VastError, match="network error"):
        Vast("k").instances()


# ---------- 26 Nov perf review: stale heartbeat, failed-host avoidance, destroy retries ----------

def test_a_stale_pulse_is_a_wedged_box():
    j = job(watch=Watch(hb_stale_min=12, setup_grace_min=40))
    hb = {"phase": "data", "step": 0, "exit_rc": None}
    assert sup.decide(obs(hb=hb, t_pulse=10_000.0 - 11 * 60), j)[0] == "continue"
    a, r = sup.decide(obs(hb=hb, t_pulse=10_000.0 - 13 * 60), j)
    assert a == "replace_failure" and "wedged" in r
    done = {"phase": "done", "step": 600, "exit_rc": 0}
    assert sup.decide(obs(hb=done, t_pulse=10_000.0 - 60 * 60), j)[0] == "done"   # exit status wins
    # a fresh pulse with no heartbeat yet (long restore) is alive, not a boot hang
    assert sup.decide(obs(now=13 * 60, t_start=0, ep_start=0, t_step=0, hb=None, t_pulse=13 * 60 - 30), j)[0] == "continue"


def _state(tmp_path, name, episodes):
    (tmp_path / f"{name}.json").write_text(json.dumps({"job": name, "spent": 0, "failures": 0, "outcome": None,
                                                       "episodes": episodes}))


def test_failed_hosts_are_avoided_for_a_day(tmp_path):
    now = 1_000_000.0
    ep = lambda offer, reason, age_h, step=0, machine=None: {"offer": offer, "machine": machine, "end": now - age_h * 3600,
                                                          "end_reason": reason, "max_step": step}
    _state(tmp_path, "a", [ep(1, "no heartbeat 12 min after renting (status loading): boot hung", 1, machine=9),
                           ep(2, "instance exited (preempted?)", 1),                    # market, not the host
                           ep(3, "no heartbeat 12 min after renting: boot hung", 30),  # too old
                           ep(4, "pulse unchanged for 12 min (phase data, step 0): box wedged", 2)])
    _state(tmp_path, "b", [ep(5, "no progress 40 min into the episode (phase data)", 3)])
    (tmp_path / "junk.json").write_text("{not json")                      # a peer's broken state: skipped
    assert sup.failed_hosts(tmp_path, "a", now=now, out=lambda m: None) == {("offer", 1), ("machine", 9), ("offer", 4)}
    # job b sees a's boot hang (global) but not a's wedge (job-scoped); its own no-progress counts
    assert sup.failed_hosts(tmp_path, "b", now=now, out=lambda m: None) == {("offer", 1), ("machine", 9), ("offer", 5)}


def test_provision_skips_failed_hosts(tmp_path):
    class V(FakeVast):
        def search_all(self, m):
            return [offer(id=7, machine_id=70), offer(id=8, machine_id=80, min_bid=0.3)]
    v = V([["running"]])
    j = job()
    p = Project(bucket="b", b2_env="", vast_env="", repo="o/r", state_dir=str(tmp_path))
    inst, rate, od, o = sup.provision(j, p, v, B2Creds("k", "s", "e"), "c", 0, lambda m: None,
                                      avoid={("machine", 70)})
    assert o["id"] == 8
    with pytest.raises(RuntimeError, match="failed in the last"):
        sup.provision(j, p, v, B2Creds("k", "s", "e"), "c", 0, lambda m: None, avoid={("offer", 7), ("offer", 8)})


def test_a_failed_destroy_is_retried_then_stops_and_is_retried_at_restart(tmp_path, monkeypatch):
    from cloudrun.vast import VastError
    monkeypatch.setattr(sup, "DESTROY_WAIT_S", 0)
    s3 = FakeS3()

    class V(FakeVast):
        fail = True

        def destroy(self, inst):
            if self.fail:
                raise VastError("DNS down")
            super().destroy(inst)
    v = V([["running"]])
    j = job(watch=Watch(poll_s=0))
    p = Project(bucket="b", b2_env="", vast_env="", repo="o/r", state_dir=str(tmp_path))

    def hb(*_a, **_k):
        raise KeyboardInterrupt
    monkeypatch.setattr(sup, "heartbeat", hb)
    out = sup.run(j, p, v, s3, B2Creds("k", "s", "e"), "c", out=lambda m: None)
    st = json.loads((tmp_path / "t.json").read_text())
    assert out.startswith("stopped") and "could not be destroyed" in out and st["episodes"][0]["destroy_pending"]
    v.fail = False
    j2 = job(budget=Budget(max_spend=1e-12, max_hours=1), watch=Watch(poll_s=0))   # stop before renting
    sup.run(j2, p, v, s3, B2Creds("k", "s", "e"), "c", out=lambda m: None)
    st = json.loads((tmp_path / "t.json").read_text())
    assert 101 in v.destroyed and not st["episodes"][0]["destroy_pending"]


def test_a_lost_rent_reply_is_adopted_by_label(tmp_path):
    from cloudrun.vast import VastError

    class V(FakeVast):
        labels = {}

        def rent(self, offer_id, body):
            inst = super().rent(offer_id, body)
            self.labels[inst] = body["label"]
            raise VastError("timeout")          # the box exists but the reply never came

        def instances(self):
            return [{"id": i, "actual_status": "running", "label": self.labels.get(i)} for i in self.live]
    v = V([["running"]])
    p = Project(bucket="b", b2_env="", vast_env="", repo="o/r", state_dir=str(tmp_path))
    st = sup.State(tmp_path / "t.json", "t")
    inst, e = sup.provision(job(), p, v, B2Creds("k", "s", "e"), "c", 0, lambda m: None, st=st)
    assert inst == 101 and st.d["episodes"][0]["instance"] == 101 and st.d["renting"] is None
    assert len(v.rented) == 1                    # did not rent a second box


def test_recover_destroys_an_unrecorded_rent_and_applies_a_saved_done(tmp_path):
    class V(FakeVast):
        def instances(self):
            return [{"id": i, "actual_status": "running", "label": "cloudrun:t:ep1:x" if i == 77 else None}
                    for i in self.live]
    v = V([])
    v.live[77] = ["running"]
    v.live[55] = ["running"]
    now = time.time()
    (tmp_path / "t.json").write_text(json.dumps({"job": "t", "spent": 0.0, "failures": 0, "outcome": None,
        "renting": {"label": "cloudrun:t:ep1:x", "offer": 7, "rate": 0.3, "od": False, "time": now - 60},
        "episodes": [{"n": 0, "instance": 55, "rate": 0.3, "start": now - 3600, "end": None, "end_reason": None,
                      "max_step": 600, "destroy_pending": True, "pending_action": "done",
                      "pending_reason": "job exited 0"}]}))
    p = Project(bucket="b", b2_env="", vast_env="", repo="o/r", state_dir=str(tmp_path))
    out = sup.run(job(watch=Watch(poll_s=0)), p, v, FakeS3(), B2Creds("k", "s", "e"), "c", out=lambda m: None)
    st = json.loads((tmp_path / "t.json").read_text())
    assert out == "done" and set(v.destroyed) == {55, 77} and v.rented == []
    assert all(e["end"] and not e["destroy_pending"] for e in st["episodes"]) and st["spent"] > 0.29


def test_a_vast_outage_still_enforces_the_spend_cap(tmp_path, monkeypatch):
    from cloudrun.vast import VastError

    class V(FakeVast):
        def poll(self, inst):
            raise VastError("DNS down")
    v = V([["running"]])
    j = job(budget=Budget(max_spend=0.5, max_hours=10), watch=Watch(poll_s=0))
    p = Project(bucket="b", b2_env="", vast_env="", repo="o/r", state_dir=str(tmp_path))
    clock = {"t": 1_000_000.0}

    def fake_time():
        clock["t"] += 3600        # an hour per call: the running cost passes the cap within a few polls
        return clock["t"]
    monkeypatch.setattr(sup.time, "time", fake_time)
    out = sup.run(j, p, v, FakeS3(), B2Creds("k", "s", "e"), "c", out=lambda m: None)
    assert out.startswith("stop") and "spend cap" in out and v.destroyed == [101]


# ---------- 26 Nov, Astra's second validation of c42068f ----------

def _p(tmp_path):
    return Project(bucket="b", b2_env="", vast_env="", repo="o/r", state_dir=str(tmp_path))


def test_one_supervisor_per_job(tmp_path):
    with sup.job_lock(tmp_path, "t"):
        with pytest.raises(SystemExit, match="another supervisor"):
            with sup.job_lock(tmp_path, "t"):
                pass
    with sup.job_lock(tmp_path, "t"):        # released on exit
        pass


def test_a_signal_as_the_rent_deferral_ends_still_destroys_the_recorded_box(tmp_path, monkeypatch):
    v = FakeVast([["running"]])
    real = sup.provision

    def prov(*a, **k):
        real(*a, **k)                        # rented and recorded in the ledger ...
        raise sup._Terminate("signal 15")    # ... but the signal lands before the caller sees the id
    monkeypatch.setattr(sup, "provision", prov)
    out = sup.run(job(watch=Watch(poll_s=0)), _p(tmp_path), v, FakeS3(), B2Creds("k", "s", "e"), "c", out=lambda m: None)
    st = json.loads((tmp_path / "t.json").read_text())
    assert out.startswith("stopped") and v.destroyed == [101] and st["episodes"][0]["end"]


def test_done_is_saved_with_the_closure_even_if_a_signal_follows(tmp_path, monkeypatch):
    s3 = FakeS3()
    v = FakeVast([["running"]])
    s3.put_object("b", "runs/t/heartbeat/ep000.json", json.dumps({"episode": 0, "step": 600, "exit_rc": 0, "phase": "done"}))
    real = sup._end_episode

    def end(*a, **k):
        real(*a, **k)
        raise sup._Terminate("signal 15")
    monkeypatch.setattr(sup, "_end_episode", end)
    sup.run(job(watch=Watch(poll_s=0)), _p(tmp_path), v, s3, B2Creds("k", "s", "e"), "c", out=lambda m: None)
    assert json.loads((tmp_path / "t.json").read_text())["outcome"] == "done"
    with pytest.raises(RuntimeError, match="already done"):      # and is never rented again
        sup.run(job(watch=Watch(poll_s=0)), _p(tmp_path), v, s3, B2Creds("k", "s", "e"), "c", out=lambda m: None)


def test_recover_closes_a_legacy_destroy_pending_and_adds_only_the_unbanked_time(tmp_path):
    v = FakeVast([])
    v.live[55] = ["running"]
    now = time.time()
    (tmp_path / "t.json").write_text(json.dumps({"job": "t", "spent": 1.0, "failures": 0, "outcome": None,
        "episodes": [{"n": 0, "instance": 55, "rate": 1.0, "start": now - 7200, "end": now - 3600,
                      "end_reason": "stalled", "max_step": 5, "destroy_pending": True}]}))
    st = sup.State(tmp_path / "t.json", "t")
    sup.recover(st, v, lambda m: None)
    assert v.destroyed == [55] and not st.d["episodes"][0]["destroy_pending"]
    assert 1.99 < st.d["spent"] < 2.01                 # 1.0 banked + the hour since, not the whole 2 h again


def test_a_recent_unlisted_rent_blocks_recovery(tmp_path):
    v = FakeVast([])
    (tmp_path / "t.json").write_text(json.dumps({"job": "t", "spent": 0.0, "failures": 0, "outcome": None,
        "renting": {"label": "cloudrun:t:ep0:x", "offer": 7, "rate": 0.3, "od": False, "time": time.time() - 60},
        "episodes": []}))
    with pytest.raises(sup.RentUnresolved):
        sup.recover(sup.State(tmp_path / "t.json", "t"), v, lambda m: None)


def test_a_pulse_failure_alone_does_not_kill_a_training_box():
    j = job(watch=Watch(hb_stale_min=12, stall_min=20))
    hb = {"phase": "run", "step": 900, "exit_rc": None}
    o = obs(hb=hb, ep_step=900, t_step=10_000.0 - 60, t_pulse=10_000.0 - 60 * 60, t_hb=10_000.0 - 60)
    assert sup.decide(o, j)[0] == "continue"
    o = obs(hb=hb, ep_step=900, t_step=10_000.0 - 13 * 60, t_pulse=10_000.0 - 13 * 60, t_hb=10_000.0 - 13 * 60)
    assert sup.decide(o, j)[0] == "replace_failure"


def test_b2_read_errors_skip_the_poll_then_a_long_outage_stops(tmp_path, monkeypatch):
    from botocore.exceptions import EndpointConnectionError
    v = FakeVast([["running"]])
    calls = {"n": 0}

    def hb(*_a, **_k):
        calls["n"] += 1
        raise EndpointConnectionError(endpoint_url="https://b2")
    monkeypatch.setattr(sup, "heartbeat", hb)
    clock = {"t": 1_000_000.0}

    def fake_time():
        clock["t"] += 60
        return clock["t"]
    monkeypatch.setattr(sup.time, "time", fake_time)
    out = sup.run(job(budget=Budget(max_spend=50, max_hours=10), watch=Watch(poll_s=0)), _p(tmp_path), v, FakeS3(),
                  B2Creds("k", "s", "e"), "c", out=lambda m: None)
    assert out.startswith("stop") and "B2 heartbeat unreadable" in out and calls["n"] > 5 and v.destroyed == [101]


def test_an_ambiguous_rent_failure_never_rents_a_second_box(tmp_path, monkeypatch):
    from cloudrun.vast import VastError
    monkeypatch.setattr(sup, "RENT_LOOKUP_WAIT_S", 0)

    class V(FakeVast):
        def search_all(self, m):
            return [offer(id=7), offer(id=8, min_bid=0.3)]

        def rent(self, offer_id, body):
            super().rent(offer_id, body)
            raise VastError("timeout")            # rented, but the list does not show its label (lag)

        def instances(self):
            return [{"id": i, "actual_status": "running", "label": None} for i in self.live]
    v = V([["running"], ["running"]])
    st = sup.State(tmp_path / "t.json", "t")
    with pytest.raises(sup.RentUnresolved):
        sup.provision(job(), _p(tmp_path), v, B2Creds("k", "s", "e"), "c", 0, lambda m: None, st=st)
    assert len(v.rented) == 1 and st.d["renting"]["offer"] == 7


def test_a_definite_rent_refusal_tries_the_next_offer(tmp_path):
    from cloudrun.vast import VastError

    class V(FakeVast):
        def search_all(self, m):
            return [offer(id=7), offer(id=8, min_bid=0.3)]

        def rent(self, offer_id, body):
            if offer_id == 7:
                raise VastError("PUT /asks/7/ -> 400: no_such_ask", code=400)
            return super().rent(offer_id, body)
    v = V([["running"]])
    st = sup.State(tmp_path / "t.json", "t")
    inst, e = sup.provision(job(), _p(tmp_path), v, B2Creds("k", "s", "e"), "c", 0, lambda m: None, st=st)
    assert e["offer"] == 8 and st.d["renting"] is None


def test_only_the_final_pass_publishes_the_exit_status(tmp_path):
    s3 = FakeS3()
    s, _run, log = _syncer(tmp_path, s3)
    log.write_text("CLOUDRUN_EXIT rc=0\n")
    assert s.once()["exit_rc"] is None
    assert s.once(final=True)["exit_rc"] == 0


def test_a_peer_state_of_the_wrong_shape_is_skipped(tmp_path):
    (tmp_path / "peer.json").write_text("[]")
    _state(tmp_path, "a", [])
    assert sup.failed_hosts(tmp_path, "a", out=lambda m: None) == set()
    (tmp_path / "a.json").write_text("[]")
    with pytest.raises(ValueError):
        sup.failed_hosts(tmp_path, "a", out=lambda m: None)


# ---------- 26 Nov, Astra's third validation of d5a5f8e ----------

def test_a_pulse_outage_never_hides_a_readable_exit(tmp_path, monkeypatch):
    from botocore.exceptions import EndpointConnectionError
    s3 = FakeS3()
    s3.put_object("b", "runs/t/heartbeat/ep000.json", json.dumps({"episode": 0, "step": 600, "exit_rc": 0, "phase": "done", "time": 1}))

    def pl(*_a, **_k):
        raise EndpointConnectionError(endpoint_url="https://b2")
    monkeypatch.setattr(sup, "pulse", pl)
    out = sup.run(job(watch=Watch(poll_s=0)), _p(tmp_path), FakeVast([["running"]]), s3, B2Creds("k", "s", "e"), "c",
                  out=lambda m: None)
    assert out == "done"


def test_charges_seen_in_the_same_poll_beat_a_done(tmp_path):
    s3 = FakeS3()
    s3.put_object("b", "runs/t/heartbeat/ep000.json", json.dumps({"episode": 0, "step": 600, "exit_rc": 0, "phase": "done", "time": 1}))
    v = FakeVast([["running"]])
    v.bw = 6.0                                   # $6 of bandwidth reported by this poll, cap $5
    out = sup.run(job(watch=Watch(poll_s=0)), _p(tmp_path), v, s3, B2Creds("k", "s", "e"), "c", out=lambda m: None)
    assert out.startswith("stop: spend cap")


def test_a_legacy_record_whose_box_is_already_gone_adds_no_spend(tmp_path):
    v = FakeVast([])
    now = time.time()
    (tmp_path / "t.json").write_text(json.dumps({"job": "t", "spent": 1.0, "failures": 0, "outcome": None,
        "episodes": [{"n": 0, "instance": 55, "rate": 1.0, "start": now - 90000, "end": now - 86400,
                      "end_reason": "job exited 0", "max_step": 5, "destroy_pending": True}]}))
    st = sup.State(tmp_path / "t.json", "t")
    assert sup.recover(st, v, lambda m: None).startswith("done")       # the legacy decision is applied
    assert st.d["spent"] == 1.0 and st.d["outcome"] == "done"


def test_a_destroy_that_succeeds_on_the_second_round_lets_the_replacement_run(tmp_path, monkeypatch):
    from cloudrun.vast import VastError
    monkeypatch.setattr(sup, "DESTROY_WAIT_S", 0)
    s3 = FakeS3()

    class V(FakeVast):
        fails = sup.DESTROY_TRIES          # the whole first round fails, the second succeeds

        def destroy(self, inst):
            if self.fails:
                self.fails -= 1
                raise VastError("DNS down")
            super().destroy(inst)
    v = V([["running", "exited"], ["running"]])
    real_hb = sup.heartbeat

    def hb(s3_, project, name, ep):
        if ep == 1:
            s3.put_object("b", f"runs/{name}/heartbeat/ep001.json",
                          json.dumps({"episode": 1, "step": 600, "exit_rc": 0, "phase": "done", "time": 1}))
        return real_hb(s3_, project, name, ep)
    monkeypatch.setattr(sup, "heartbeat", hb)
    out = sup.run(job(watch=Watch(poll_s=0)), _p(tmp_path), v, s3, B2Creds("k", "s", "e"), "c", out=lambda m: None)
    assert out == "done" and len(v.rented) == 2


def test_an_explicit_refusal_in_a_2xx_reply_is_definite():
    from cloudrun.vast import Vast, VastError

    class V(Vast):
        def call(self, *a, **k):
            return {"success": False, "error": "no_such_ask", "msg": "gone"}
    with pytest.raises(VastError) as ei:
        V("k").rent(7, {})
    assert ei.value.definite

    class W(Vast):
        def call(self, *a, **k):
            return {"weird": True}
    with pytest.raises(VastError) as ei:
        W("k").rent(7, {})
    assert not ei.value.definite


def test_a_late_listed_rent_is_destroyed_before_the_next_rent(tmp_path):
    v = FakeVast([["running"]])
    v.live[77] = ["running"]

    class V(FakeVast):
        pass
    now = time.time()
    st = sup.State(tmp_path / "t.json", "t")
    st.d["late_labels"] = [{"label": "cloudrun:t:ep0:x", "offer": 7, "rate": 0.3, "od": False, "time": now - 3600}]
    v.instances = lambda: [{"id": i, "actual_status": "running", "label": "cloudrun:t:ep0:x" if i == 77 else None}
                           for i in v.live]
    inst, e = sup.provision(job(), _p(tmp_path), v, B2Creds("k", "s", "e"), "c", 0, lambda m: None, st=st)
    assert 77 in v.destroyed and e["n"] == 1 and st.d["late_labels"] == [] and st.d["episodes"][0]["end"]


def test_the_summary_reports_the_persisted_outcome(tmp_path, monkeypatch):
    s3 = FakeS3()
    s3.put_object("b", "runs/t/heartbeat/ep000.json", json.dumps({"episode": 0, "step": 600, "exit_rc": 0, "phase": "done", "time": 1}))
    real = sup._end_episode
    msgs = []

    def end(*a, **k):
        real(*a, **k)
        raise sup._Terminate("signal 15")
    monkeypatch.setattr(sup, "_end_episode", end)
    sup.run(job(watch=Watch(poll_s=0)), _p(tmp_path), FakeVast([["running"]]), s3, B2Creds("k", "s", "e"), "c", out=msgs.append)
    assert any(m.startswith("t: done;") for m in msgs)


def test_a_peer_episode_with_a_bad_end_is_skipped(tmp_path):
    _state(tmp_path, "peer", [{"n": 0, "end": "yesterday", "end_reason": "boot hung"}])
    _state(tmp_path, "a", [])
    assert sup.failed_hosts(tmp_path, "a", out=lambda m: None) == set()


# ---------- 26 Nov, Astra's fourth validation of 107d7ff ----------

def test_an_unreadable_pulse_is_no_evidence_of_a_boot_hang_or_wedge():
    j = job(watch=Watch(hb_stale_min=12, boot_min=12, setup_grace_min=40))
    o = obs(now=13 * 60, t_start=0, ep_start=0, t_step=0, hb=None, t_pulse=None, pulse_unread=True)
    assert sup.decide(o, j)[0] == "continue"
    o = obs(hb={"phase": "data", "step": 0, "exit_rc": None}, t_pulse=10_000.0 - 30 * 60, t_hb=10_000.0 - 30 * 60,
            pulse_unread=True)
    assert sup.decide(o, j)[0] == "continue"


def test_only_allowlisted_refusals_are_definite():
    from cloudrun.vast import Vast, VastError

    class V(Vast):
        def call(self, *a, **k):
            return {"success": False, "error": "upstream_timeout"}
    with pytest.raises(VastError) as ei:
        V("k").rent(7, {})
    assert not ei.value.definite


def test_late_adoption_uses_the_billed_rate_and_clears_the_pending_rent(tmp_path):
    v = FakeVast([])
    v.live[77] = ["running"]
    v.instances = lambda: [{"id": 77, "actual_status": "running", "label": "L", "dph_total": 0.9,
                            "inet_down_billed": 0, "inet_up_billed": 0}]
    st = sup.State(tmp_path / "t.json", "t")
    st.d["renting"] = {"label": "L", "offer": 7, "rate": 0.3, "od": False, "time": time.time() - 3600, "ep": 0}
    new = sup._adopt_late(st, v, lambda m: None)
    assert new[0]["rate"] == 0.9 and st.d["renting"] is None
    assert json.loads((tmp_path / "t.json").read_text())["renting"] is None


def test_episode_numbers_are_never_reused_after_an_unresolved_rent(tmp_path):
    v = FakeVast([["running"]])
    st = sup.State(tmp_path / "t.json", "t")
    st.d["next_ep"] = 3                  # ep0-2 were attempted (one lost) before
    inst, e = sup.provision(job(), _p(tmp_path), v, B2Creds("k", "s", "e"), "c", 0, lambda m: None, st=st)
    assert e["n"] == 3 and v.rented[0][1]["env"]["CLOUDRUN_EPISODE"] == "3" and st.d["next_ep"] == 4


def test_a_listing_error_before_renting_retries_instead_of_stopping(tmp_path, monkeypatch):
    from cloudrun.vast import VastError
    monkeypatch.setattr(sup, "NO_OFFER_WAIT_S", 0)
    s3 = FakeS3()

    class V(FakeVast):
        n = 0

        def search_all(self, m):
            self.n += 1
            if self.n == 1:
                raise VastError("DNS down")
            return super().search_all(m)
    s3.put_object("b", "runs/t/heartbeat/ep000.json", json.dumps({"episode": 0, "step": 600, "exit_rc": 0, "phase": "done", "time": 1}))
    out = sup.run(job(watch=Watch(poll_s=0)), _p(tmp_path), V([["running"]]), s3, B2Creds("k", "s", "e"), "c", out=lambda m: None)
    assert out == "done"


def test_malformed_host_ids_are_ignored(tmp_path):
    _state(tmp_path, "peer", [{"n": 0, "offer": 5, "machine": [], "end": time.time(), "end_reason": "boot hung"}])
    _state(tmp_path, "a", [])
    assert sup.failed_hosts(tmp_path, "a", out=lambda m: None) == {("offer", 5)}


def test_a_slow_host_is_replaced_once_the_window_has_passed():
    j = job(watch=Watch(min_steps_per_min=300, slow_window_min=10, stall_min=20))
    hb = {"phase": "run", "step": 21150, "exit_rc": None}
    slow = obs(hb=hb, ep_step=21150, t_step=10_000.0 - 30, first_step=(10_000.0 - 11 * 60, 20_600))
    a, r = sup.decide(slow, j)
    assert a == "replace_failure" and "slow host" in r and "no progress" in r      # failed_hosts avoids it
    fast = obs(hb=hb, ep_step=27_000, t_step=10_000.0 - 30, first_step=(10_000.0 - 11 * 60, 20_600))
    assert sup.decide(fast, j)[0] == "continue"
    early = obs(hb=hb, ep_step=20_700, t_step=10_000.0 - 30, first_step=(10_000.0 - 5 * 60, 20_600))
    assert sup.decide(early, j)[0] == "continue"                                   # window not yet passed
    assert sup.decide(slow, job(watch=Watch(stall_min=20)))[0] == "continue"       # off by default
