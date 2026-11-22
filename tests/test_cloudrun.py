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
        assert j.budget.max_spend <= 10, f"{f.name}: test-run budget unexpectedly large"


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
