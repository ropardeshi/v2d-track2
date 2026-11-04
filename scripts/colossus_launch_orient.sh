#!/usr/bin/env bash
# Retrain the 3 cheapest near-miss T1 episodes with object-orientation reward.
# GPU 0 is never used.
set -euo pipefail
HOST_ROOT="${HOST_ROOT:-/colossus/isaac-automation}"
V2D="$HOST_ROOT/v2d_track2"
ROOT="$V2D/outputs/tier1_orient"
mkdir -p "$ROOT"
chmod +x "$V2D/colossus_start_gpu.sh" "$V2D/run_tier1_350m_incontainer.py"

# GPU 1 = trash can (3.1 cm, 1 mm from MP)
# GPU 2 = book (28.8 deg, 0.2 deg from SP)
# GPU 3 = woven basket (3.0 cm / 22 deg, on the MP line)
declare -A SHARDS
SHARDS[1]=18
SHARDS[2]=11
SHARDS[3]=27

echo "$(date '+%Y-%m-%d %H:%M:%S') launching orientation near-miss shards" | tee -a "$ROOT/colossus_launch.log"
for gpu in 1 2 3; do
  eps="${SHARDS[$gpu]}"
  name="flash_chord-latest-gpu${gpu}"
  echo "GPU $gpu -> episode $eps container $name" | tee -a "$ROOT/colossus_launch.log"
  "$V2D/colossus_start_gpu.sh" "$gpu"
  docker exec -d -w /workspace "$name" \
    bash -c "python -u /home/horde/isaac-automation/v2d_track2/run_tier1_350m_incontainer.py --episodes ${eps} --root /home/horde/isaac-automation/v2d_track2/outputs/tier1_orient --curriculum recon_body_flash_sac_orient --object-orientation-std 0.25 --label gpu${gpu} >> /home/horde/isaac-automation/v2d_track2/outputs/tier1_orient/gpu${gpu}.out 2>&1"
done
sleep 3
for gpu in 1 2 3; do
  echo "== gpu${gpu} ==" | tee -a "$ROOT/colossus_launch.log"
  tail -n 4 "$ROOT/gpu${gpu}.out" 2>/dev/null | tee -a "$ROOT/colossus_launch.log" || true
done
echo launched | tee -a "$ROOT/colossus_launch.log"
