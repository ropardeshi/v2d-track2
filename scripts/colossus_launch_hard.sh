#!/usr/bin/env bash
# Retrain the 7 worst remaining T1 episodes with stronger object tracking.
# GPU 0 is never used.
set -euo pipefail
HOST_ROOT="${HOST_ROOT:-/colossus/isaac-automation}"
V2D="$HOST_ROOT/v2d_track2"
ROOT="$V2D/outputs/tier1_hard"
mkdir -p "$ROOT"
chmod +x "$V2D/colossus_start_gpu.sh" "$V2D/run_tier1_350m_incontainer.py"

# One disaster episode per GPU (highest leftover MPPE after the best-of mix).
declare -A SHARDS
SHARDS[1]=4
SHARDS[2]=25
SHARDS[3]=28
SHARDS[4]=1
SHARDS[5]=2
SHARDS[6]=13
SHARDS[7]=21

echo "$(date '+%Y-%m-%d %H:%M:%S') launching hard-episode shards" | tee -a "$ROOT/colossus_launch.log"
for gpu in 1 2 3 4 5 6 7; do
  eps="${SHARDS[$gpu]}"
  name="flash_chord-latest-gpu${gpu}"
  echo "GPU $gpu -> episode $eps container $name" | tee -a "$ROOT/colossus_launch.log"
  "$V2D/colossus_start_gpu.sh" "$gpu"
  docker exec -d -w /workspace "$name" \
    bash -c "python -u /home/horde/isaac-automation/v2d_track2/run_tier1_350m_incontainer.py --episodes ${eps} --root /home/horde/isaac-automation/v2d_track2/outputs/tier1_hard --curriculum recon_body_flash_sac_object --object-position-std 0.1 --label gpu${gpu} >> /home/horde/isaac-automation/v2d_track2/outputs/tier1_hard/gpu${gpu}.out 2>&1"
done
sleep 3
for gpu in 1 2 3 4 5 6 7; do
  echo "== gpu${gpu} ==" | tee -a "$ROOT/colossus_launch.log"
  tail -n 4 "$ROOT/gpu${gpu}.out" 2>/dev/null | tee -a "$ROOT/colossus_launch.log" || true
done
echo launched | tee -a "$ROOT/colossus_launch.log"
