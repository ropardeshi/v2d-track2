#!/usr/bin/env bash
# Start a detached FlashCHORD container pinned to one GPU (1-7). Never GPU 0.
set -euo pipefail

GPU="${1:?gpu index 1-7}"
if [[ "$GPU" == "0" ]]; then
  echo "refusing GPU 0 (reserved on colossus)" >&2
  exit 1
fi

HOST_ROOT="${HOST_ROOT:-/colossus/isaac-automation}"
FLASH="$HOST_ROOT/video_to_data/robotic_grounding/flash_chord"
ASSETS="$HOST_ROOT/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets"
V2D="$HOST_ROOT/v2d_track2"
IMAGE="${IMAGE:-flash_chord:latest}"
NAME="flash_chord-latest-gpu${GPU}"

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_USER="$(id -un)"
HOST_GROUP="$(id -gn)"
IDMAP_DIR="${HOME}/.cache/flash_chord/${NAME}"
mkdir -p "$IDMAP_DIR"
printf 'root:x:0:0:root:/root:/bin/bash\n%s:x:%s:%s:%s:/tmp:/bin/bash\n' \
  "$HOST_USER" "$HOST_UID" "$HOST_GID" "$HOST_USER" > "$IDMAP_DIR/passwd"
printf 'root:x:0:\n%s:x:%s:\n' "$HOST_GROUP" "$HOST_GID" > "$IDMAP_DIR/group"
chmod 0644 "$IDMAP_DIR/passwd" "$IDMAP_DIR/group"

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "already running $NAME"
  exit 0
fi

# Drop a leftover stopped container of the same name (should not happen with --rm).
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -dit --rm \
  --runtime=nvidia --gpus "device=${GPU}" \
  --name "$NAME" \
  --network host \
  --shm-size=8g \
  --user "${HOST_UID}:${HOST_GID}" \
  -e HOME=/tmp \
  -e "USER=${HOST_USER}" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -v "${ASSETS}:/v2d/assets:ro" \
  -v "${FLASH}:/workspace" \
  -v "${V2D}:/home/horde/isaac-automation/v2d_track2" \
  -v "${IDMAP_DIR}/passwd:/etc/passwd:ro" \
  -v "${IDMAP_DIR}/group:/etc/group:ro" \
  --entrypoint /bin/bash \
  "$IMAGE"
echo "started $NAME on GPU $GPU"
