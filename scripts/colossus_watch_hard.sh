#!/usr/bin/env bash
# Keep the 7 hard-episode ranking trains alive on GPUs 1-7. Never GPU 0.
# Exit when every episode has a final policy and eval parquet.
set -u
HOST_ROOT="${HOST_ROOT:-/colossus/isaac-automation}"
V2D="$HOST_ROOT/v2d_track2"
ROOT="$V2D/outputs/tier1_hard"
LOG="$ROOT/watchdog.log"
mkdir -p "$ROOT"
chmod +x "$V2D/colossus_start_gpu.sh" "$V2D/run_tier1_350m_incontainer.py"

declare -A SHARDS
SHARDS[1]=4
SHARDS[2]=25
SHARDS[3]=28
SHARDS[4]=1
SHARDS[5]=2
SHARDS[6]=13
SHARDS[7]=21

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$LOG"; }

ep_dir() {
  local ep="$1"
  python3 - "$ep" "$ROOT" <<'PY'
import sys
from pathlib import Path
ep = int(sys.argv[1])
root = Path(sys.argv[2])
matches = sorted(root.glob(f"ep{ep:02d}_*"))
print(matches[0] if matches else "")
PY
}

all_done() {
  local gpu ep dir
  for gpu in 1 2 3 4 5 6 7; do
    ep="${SHARDS[$gpu]}"
    dir="$(ep_dir "$ep")"
    [[ -n "$dir" ]] || return 1
    [[ -f "$dir/train/policy_349999104.safetensors" ]] || return 1
    ls "$dir"/eval/episode_*.parquet >/dev/null 2>&1 || return 1
  done
  return 0
}

restart_gpu() {
  local gpu="$1" ep="$2" name="flash_chord-latest-gpu${gpu}"
  log "RELAUNCH gpu${gpu} episode ${ep}"
  "$V2D/colossus_start_gpu.sh" "$gpu"
  docker exec -d -w /workspace "$name" \
    bash -c "python -u /home/horde/isaac-automation/v2d_track2/run_tier1_350m_incontainer.py --episodes ${ep} --root /home/horde/isaac-automation/v2d_track2/outputs/tier1_hard --curriculum recon_body_flash_sac_object --object-position-std 0.1 --label gpu${gpu} >> /home/horde/isaac-automation/v2d_track2/outputs/tier1_hard/gpu${gpu}.out 2>&1"
}

log "watchdog start"
while true; do
  if all_done; then
    log "ALL_EVAL_DONE"
    echo "ALL_EVAL_DONE"
    exit 0
  fi
  for gpu in 1 2 3 4 5 6 7; do
    ep="${SHARDS[$gpu]}"
    name="flash_chord-latest-gpu${gpu}"
    dir="$(ep_dir "$ep")"
    if [[ -n "$dir" && -f "$dir/train/policy_349999104.safetensors" ]] && ls "$dir"/eval/episode_*.parquet >/dev/null 2>&1; then
      continue
    fi
    if docker ps --format '{{.Names}}' | grep -qx "$name"; then
      continue
    fi
    restart_gpu "$gpu" "$ep"
  done
  sleep 60
done
