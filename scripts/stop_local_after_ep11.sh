#!/usr/bin/env bash
# Let the in-flight ep11 train+eval finish, then stop the local 350M driver
# so episodes 12-29 are not started on GPU 0 (colossus owns those).
set -u
ROOT=/home/horde/isaac-automation/v2d_track2/outputs/tier1_350m
LOG=$ROOT/train_eval.log
PIDFILE=$ROOT/driver.pid
OUT=$ROOT/watchdog.out
touch "$OUT"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$OUT"; }

log "watchdog start pid=$$"

while true; do
  if grep -qE 'ep11 EVAL_OK|ep11 TRAIN_FAIL|ep11 EVAL_FAIL' "$LOG"; then
    log "ep11 terminal line seen"
    break
  fi
  if grep -q 'ep12 TRAIN ' "$LOG"; then
    log "ep12 TRAIN started locally; stopping so colossus can own it"
    break
  fi
  sleep 10
done

pid=$(cat "$PIDFILE" 2>/dev/null || true)
log "stopping local driver pid=${pid:-none}"
if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in 1 2 3 4 5 6; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
fi

if grep -q 'ep12 TRAIN ' "$LOG"; then
  docker exec flash_chord-latest-gpu0 pkill -f 'scripts/train_flash_sac.py' 2>/dev/null || true
  log "signalled in-container train (ep12 abort if it had started)"
fi

log "watchdog done"
touch "$ROOT/local_stopped_after_ep11"
