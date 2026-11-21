#!/usr/bin/env bash
# cloudrun instance entrypoint. Runs on EVERY episode (the first box and each replacement), so it is
# idempotent: restore the run dir from B2, start the syncer, set up, pull data, run, report, flush.
#
# Started by the supervisor's onstart after it fetched the code tarball into $WORK/code. Environment:
#   CLOUDRUN_JOB CLOUDRUN_EPISODE CLOUDRUN_BUCKET CLOUDRUN_COMMIT
#   CLOUDRUN_{COMMAND,SETUP,DATA,PROGRESS_RE}_B64 CLOUDRUN_ROLLING CLOUDRUN_KEEP_ROLLING CLOUDRUN_B2_{KEY_ID,APPLICATION_KEY,ENDPOINT}
#
# Status leaves the box only through B2 (heartbeat + log tail, see sync.py). The single source of truth for
# how the job ended is the "CLOUDRUN_EXIT rc=N" line this script prints; nothing infers failure from prose.
#   rc 90 setup failed, 91 data pull failed, 92 run-dir restore failed, otherwise the command's own exit code.

set -uo pipefail
: "${WORK:?}" "${CLOUDRUN_JOB:?}" "${CLOUDRUN_COMMAND_B64:?}" "${CLOUDRUN_PROGRESS_RE_B64:?}" "${CLOUDRUN_BUCKET:?}"
export RUN_DIR=$WORK/run DATA=$WORK/data LOG=$WORK/job.log
export PYTHONPATH=$WORK/code/src${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
# multi-line / quote-heavy values travel base64-encoded (container env vars are single-line)
b64() { printf '%s' "${1:-}" | base64 -d; }
CLOUDRUN_COMMAND=$(b64 "$CLOUDRUN_COMMAND_B64"); CLOUDRUN_SETUP=$(b64 "${CLOUDRUN_SETUP_B64:-}")
CLOUDRUN_DATA=$(b64 "${CLOUDRUN_DATA_B64:-}"); export CLOUDRUN_PROGRESS_RE=$(b64 "$CLOUDRUN_PROGRESS_RE_B64")
mkdir -p "$RUN_DIR" "$DATA"
cd "$WORK"
SYNC_PID=""

phase() { echo "CLOUDRUN_PHASE $1  $(date -u +%FT%TZ)"; }

finish() {
  local rc=$1
  echo "CLOUDRUN_EXIT rc=$rc"
  [ -n "$SYNC_PID" ] && kill "$SYNC_PID" 2>/dev/null && wait "$SYNC_PID" 2>/dev/null
  # final synchronous pass so the last checkpoint and the exit status reach B2 before the box goes away
  for i in 1 2 3; do
    python -m cloudrun.sync --run "$RUN_DIR" --log "$LOG" --once --final && break
    echo "final sync attempt $i failed"; sleep 10
  done
  exit "$rc"
}

phase env
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"
nproc; free -g | head -2; df -h "$WORK" | tail -1

phase restore
python -m cloudrun.transfer down --prefix "${CLOUDRUN_ROOT:+$CLOUDRUN_ROOT/}runs/$CLOUDRUN_JOB/run" --local "$RUN_DIR" --allow-empty || finish 92
ls -la "$RUN_DIR" | tail -n +2 | head -20

python -m cloudrun.sync --run "$RUN_DIR" --log "$LOG" --interval "${CLOUDRUN_SYNC_S:-60}" > "$WORK/sync.log" 2>&1 &
SYNC_PID=$!
echo "syncer pid $SYNC_PID"

phase setup
while IFS= read -r line; do
  [ -z "$line" ] && continue
  echo "+ $line"
  bash -c "$line" < /dev/null || { echo "setup step failed: $line"; finish 90; }
done <<< "${CLOUDRUN_SETUP:-}"

phase data
# CLOUDRUN_DATA: one "remote_prefix local_dir" pair per line (remote already includes the root); local_dir under $DATA
while read -r remote local; do
  [ -z "${remote:-}" ] && continue
  python -m cloudrun.transfer down --prefix "$remote" --local "$DATA/$local" < /dev/null || { echo "data pull failed: $remote"; finish 91; }
done <<< "${CLOUDRUN_DATA:-}"
du -sh "$DATA"/* 2>/dev/null

phase run
cd "$WORK/code"
echo "commit ${CLOUDRUN_COMMIT:-?}"
echo "+ $CLOUDRUN_COMMAND"
bash -c "$CLOUDRUN_COMMAND" < /dev/null
RC=$?
phase done
finish "$RC"
