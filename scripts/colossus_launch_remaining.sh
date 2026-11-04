#!/usr/bin/env bash
# Launch remaining T1 350M episodes (12-29) across colossus GPUs 1-7.
set -euo pipefail
HOST_ROOT="${HOST_ROOT:-/colossus/isaac-automation}"
V2D="$HOST_ROOT/v2d_track2"
ROOT="$V2D/outputs/tier1_350m"
mkdir -p "$ROOT"

chmod +x "$V2D/colossus_start_gpu.sh" "$V2D/run_tier1_350m_incontainer.py"

# 18 remaining episodes, round-robin onto GPUs 1-7.
# GPU1: 12,19,26  GPU2: 13,20,27  ... GPU7: 18,25
declare -A SHARDS
for ep in $(seq 12 29); do
  gpu=$((1 + (ep - 12) % 7))
  if [[ -n "${SHARDS[$gpu]:-}" ]]; then
    SHARDS[$gpu]="${SHARDS[$gpu]},${ep}"
  else
    SHARDS[$gpu]="$ep"
  fi
done

echo "$(date '+%Y-%m-%d %H:%M:%S') launching shards" | tee -a "$ROOT/colossus_launch.log"
for gpu in 1 2 3 4 5 6 7; do
  eps="${SHARDS[$gpu]}"
  name="flash_chord-latest-gpu${gpu}"
  echo "GPU $gpu -> episodes $eps container $name" | tee -a "$ROOT/colossus_launch.log"
  "$V2D/colossus_start_gpu.sh" "$gpu"
  docker exec -d -w /workspace "$name" \
    bash -c "python -u /home/horde/isaac-automation/v2d_track2/run_tier1_350m_incontainer.py --episodes ${eps} --label gpu${gpu} >> /home/horde/isaac-automation/v2d_track2/outputs/tier1_350m/gpu${gpu}.out 2>&1"
done

sleep 3
echo "--- containers ---" | tee -a "$ROOT/colossus_launch.log"
docker ps --format 'table {{.Names}}\t{{.Status}}' | tee -a "$ROOT/colossus_launch.log"
echo "--- shard logs ---" | tee -a "$ROOT/colossus_launch.log"
for gpu in 1 2 3 4 5 6 7; do
  echo "== gpu${gpu} ==" | tee -a "$ROOT/colossus_launch.log"
  tail -n 5 "$ROOT/gpu${gpu}.out" 2>/dev/null | tee -a "$ROOT/colossus_launch.log" || true
done
echo "launched" | tee -a "$ROOT/colossus_launch.log"
