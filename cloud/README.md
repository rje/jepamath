# Cloud runs (cloudrun)

Preemptible Vast.ai GPUs with Backblaze B2 as the only durable storage. The framework is `src/cloudrun/`
(project-agnostic, extracted from microlab's `vast_run.py` / `vast_supervisor.py` / `b2_*sync.py` /
`cloud_train.sh`); this directory is the jepamath setup: `cloudrun.toml` (project) and `jobs/*.toml`.

All commands run from the repo root.

```bash
python -m cloudrun offers trackb-smoke          # live offers that fit the job (read-only)
python -m cloudrun plan trackb-smoke            # every pre-rental check; rents nothing
python -m cloudrun launch trackb-smoke --yes    # RENTS: supervise until done / cap / failure
python -m cloudrun status trackb-smoke          # spend, episodes, newest heartbeat
python -m cloudrun logs trackb-smoke            # shipped log tail of the newest episode
python -m cloudrun pull trackb-smoke /mnt/data/jepamath/trackb/cloud-smoke
python -m cloudrun instances                    # anything alive on the account
python -m cloudrun destroy <id>
```

## How a job runs

1. `launch` requires a clean, pushed HEAD: the instance downloads exactly that commit from
   github.com/rje/jepamath (public, no credentials on the box for code).
2. The supervisor ranks offers by the expected cost of one episode: the rate it would actually pay
   (interruptible bid = floor × 1.25, capped at `max_price_per_gpu`; on-demand if within 10%) ×
   `episode_hours`, plus the host's per-GB bandwidth price × (data + run dir pulled, `upload_gb` pushed).
   Bandwidth matters: at $0.036/GB the 27 GB pull alone is ~$1, more than hours of an A5000. Spend
   (for `max_spend`) is the billed `dph_total` × hours plus the instance's billed bandwidth.
3. On the box, `bootstrap.sh`: restore `runs/<job>/run/` from B2 → start the syncer → setup →
   pull `[[data]]` → run `command` → print `CLOUDRUN_EXIT rc=N` → final flush.
4. The syncer mirrors `$RUN_DIR` to B2 every 60 s (stable files only), prunes `ckpt_*.pt` beyond the
   newest 2, and writes `heartbeat/epNNN.json` (phase, step from `progress_regex`, exit code).
5. The supervisor polls the heartbeat and the instance: exit 0 → done; preempted → re-rent (the next
   box resumes from B2 via `--auto-resume`); stall / setup timeout / crash → re-rent, counted as a
   failure (`max_failures`); failure before any progress → stop. `max_spend` (banked + running episode)
   and `max_hours` are hard caps. Every exit path destroys the box and verifies it is gone.

## B2 layout (bucket `microlab-checkpoints`, prefix `jepamath/`)

```
jepamath/data/v2_128/train/possible/shard-*.tar   possible clips at 128 px (python -m jepatrain.prep_cloud)
jepamath/data/heldout/v2_triples2000_128.pt(.ids.json)
jepamath/runs/<job>/run/...                       the job's run dir (checkpoints, log.jsonl, evals)
jepamath/runs/<job>/logs/epNNN.log                log tail per episode
jepamath/runs/<job>/heartbeat/epNNN.json
```

The B2 key is microlab's checkpoint key, scoped to that one bucket (which already deletes superseded
versions). A dedicated bucket needs a key that can create buckets: make one in the B2 console, then set
`bucket`, `b2_env` and `root` in `cloudrun.toml` (and `python -m cloudrun make-bucket` if the key allows).

## Not covered

- If the supervisor process itself dies (machine reboot), the box keeps billing until the supervisor is
  restarted (it destroys orphaned episodes on start) or it is destroyed by hand. Run it under `nohup`
  and check `python -m cloudrun instances` after any local crash.
- The instance holds the B2 key in its environment; the host operator can read it.

## Cost rules (26 Nov perf review)

Of the $112.59 spent over 215 episodes, $14.84 went to boxes that never trained. Of that, 43 were boot hangs
($9.02), and 11 of those came straight after a hang on the same offer. Rules since then:

- **Failed hosts are skipped for 24 h.** The supervisor reads every job's state and does not rent an offer or
  machine that hung at boot, made no progress within the setup grace, or wedged
  (`supervisor.failed_hosts`). Preemptions don't count, since they are the market rather than the host.
- **Liveness pulse.** onstart starts `cloudrun.pulse` right after the code fetch. It writes
  `runs/<job>/pulse/epNNN.json` every minute for the life of the box, independent of uploads and of the job's
  phase. A pulse that has been seen and then stays unchanged for `watch.hb_stale_min` (12) means the box is
  frozen or offline. A box with neither heartbeat nor pulse after `boot_min` hung at boot.
- **Host cooldown.** Boot hangs exclude the host for every job. Wedges and no-progress timeouts exclude it for
  the same job only, since a slow dataset or a B2 outage can cause them on a healthy host.
- **Destroy is retried.** It gets 6 tries, 30 s apart, with signals deferred. The decision is saved before
  the first DELETE. If all tries fail, the episode stays open (still billing, still accounted), and the next
  launch closes it before any precheck (`supervisor.recover`). It then applies the saved decision, so a job
  that had finished is not rerun.
- **Rents are labelled and recorded before the request.** A rent whose reply is lost is found by its label and
  adopted rather than doubled. If a crash happens in between, the next launch destroys the box.
- **Vast API outages no longer suspend the caps.** Spend and wall clock are judged from local time and the last
  known rate.
- **Inline evals.** Endpoint-focused 30k jobs pass `--eval-every 15000`, which evaluates at 0, 15k and 30k and
  saves weights there. That saves 8–11 min per run. The 500-step recovery checkpoints are unchanged. The evals
  use no RNG, so this does not change the training trajectory. Keep 2500 only where the eval curve is the
  measurement.
