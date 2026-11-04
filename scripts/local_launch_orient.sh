#!/usr/bin/env bash
# Sequential orientation retrains on the idle local FlashCHORD container (GPU 0).
# Order is cheapest remaining SP/MP flips. Stop starting new trains before freeze.
set -u
V2D="/home/horde/isaac-automation/v2d_track2"
ROOT="$V2D/outputs/tier1_orient"
NAME="flash_chord-latest-gpu0"
LOG="$ROOT/driver.log"
# Last new train start: 3h before 5pm EST freeze (pack + upload buffer).
DEADLINE=$(date -d '2026-11-04 19:00 UTC' +%s)
ORDER=(11 27 26 18 20 3)

mkdir -p "$ROOT"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  log "ERROR container $NAME is not running"
  exit 1
fi

log "orient driver start order=${ORDER[*]} deadline=$(date -d @$DEADLINE -u '+%Y-%m-%d %H:%M UTC')"
for ep in "${ORDER[@]}"; do
  now=$(date +%s)
  if (( now >= DEADLINE )); then
    log "DEADLINE skip remaining from ep${ep}"
    break
  fi
  log "START ep${ep}"
  docker exec -w /workspace "$NAME" \
    python -u /home/horde/isaac-automation/v2d_track2/run_tier1_350m_incontainer.py \
      --episodes "${ep}" \
      --root /home/horde/isaac-automation/v2d_track2/outputs/tier1_orient \
      --curriculum recon_body_flash_sac_orient \
      --object-orientation-std 0.25 \
      --label "ep${ep}" >> "$ROOT/ep${ep}.out" 2>&1
  rc=$?
  log "END ep${ep} rc=${rc}"
  /home/horde/isaac-automation/v2d_track2/.venv/bin/python \
    "$V2D/mix_and_submit_if_better.py" >> "$ROOT/mix.log" 2>&1 || log "mix failed ep${ep}"
done
log "orient driver done"
echo "ALL_ORIENT_DONE"
