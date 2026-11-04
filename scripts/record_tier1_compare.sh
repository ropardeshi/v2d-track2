#!/usr/bin/env bash
# Record G1 policy MP4s for all 30 T1 episodes and stack them beside the
# original exo-camera clip. Uses local GPU 0 only (colossus 1-7 keep training).
set -u
CONTAINER="${CONTAINER:-flash_chord-latest-gpu0}"
MAP="${MAP:-/home/horde/isaac-automation/v2d_track2/episode_map.json}"
POLICY_ROOT="${POLICY_ROOT:-/home/horde/isaac-automation/v2d_track2/outputs/tier1_350m}"
HUMAN_DIR="${HUMAN_DIR:-/home/horde/isaac-automation/video_to_data_challenge/track_2/tier_1_multiview_caption/videos/chunk-000/observation.images.exo_camera}"
OUT="${OUT:-/home/horde/isaac-automation/v2d_track2/outputs/tier1_compare}"
LOG="$OUT/record.log"
mkdir -p "$OUT"

python3 - "$MAP" "$POLICY_ROOT" "$HUMAN_DIR" "$OUT" "$CONTAINER" "$LOG" <<'PY'
import json, subprocess, sys, time
from pathlib import Path

map_path, policy_root, human_dir, out, container, log_path = sys.argv[1:]
policy_root = Path(policy_root)
human_dir = Path(human_dir)
out = Path(out)
log = Path(log_path)
episodes = json.loads(Path(map_path).read_text())["episodes"]


def log_line(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with log.open("a") as fh:
        fh.write(line + "\n")


def docker(cmd: list[str], logfile: Path) -> int:
    full = [
        "docker", "exec", "-w", "/workspace", container,
        "xvfb-run", "-a", "-s", "-screen 0 1280x720x24",
        "env", "PYTHONUNBUFFERED=1", "MUJOCO_GL=egl", "PYOPENGL_PLATFORM=egl",
        "PYGLET_HEADLESS=1", *cmd,
    ]
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a") as fh:
        fh.write("\n$ " + " ".join(full) + "\n")
        fh.flush()
        return subprocess.run(full, stdout=fh, stderr=subprocess.STDOUT).returncode


fail = 0
for rec in episodes:
    ep = int(rec["episode_index"])
    seq = rec["sequence_id"]
    frames = int(rec["frames"])
    ep_dir = out / f"ep{ep:02d}_{seq}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    human_src = human_dir / f"episode_{ep:06d}.mp4"
    human_dst = ep_dir / "human.mp4"
    policy_mp4 = ep_dir / "policy.mp4"
    compare_mp4 = ep_dir / "compare.mp4"
    rec_log = ep_dir / "record.log"
    ckpts = list(policy_root.glob(f"ep{ep:02d}_*/train/policy_349999104.safetensors"))
    if not ckpts:
        log_line(f"ep{ep:02d} SKIP no 350M policy")
        fail += 1
        continue
    if not human_src.is_file():
        log_line(f"ep{ep:02d} SKIP no human clip {human_src}")
        fail += 1
        continue
    if not human_dst.exists():
        human_dst.symlink_to(human_src)

    if not policy_mp4.is_file() or policy_mp4.stat().st_size < 10_000:
        log_line(f"ep{ep:02d} RECORD {seq} frames={frames}")
        rc = docker(
            [
                "python", "-u", "scripts/view_policy.py",
                f"evaluation.checkpoint={ckpts[0]}",
                "evaluation.reset_mode=explicit",
                "scene.world_count=1",
                "viewer.backend=mp4",
                f"viewer.output_path={policy_mp4}",
                f"viewer.num_frames={frames}",
                "viewer.video_fps=30",
                "viewer.width=1280",
                "viewer.height=720",
                "viewer.realtime=false",
                "markers.enabled=false",
                "logging.mode=disabled",
                f"hydra.run.dir={ep_dir}/hydra",
            ],
            rec_log,
        )
        if rc != 0 or not policy_mp4.is_file():
            log_line(f"ep{ep:02d} RECORD_FAIL rc={rc}")
            fail += 1
            continue
        log_line(f"ep{ep:02d} RECORD_OK {policy_mp4}")
    else:
        log_line(f"ep{ep:02d} RECORD_SKIP {policy_mp4}")

    if not compare_mp4.is_file() or compare_mp4.stat().st_size < 10_000:
        cmp_log = ep_dir / "ffmpeg.log"
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(human_dst), "-i", str(policy_mp4),
            "-filter_complex",
            "[0:v]scale=960:720:force_original_aspect_ratio=decrease,"
            "pad=960:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[h];"
            "[1:v]scale=960:720:force_original_aspect_ratio=decrease,"
            "pad=960:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[p];"
            "[h][p]hstack=inputs=2[v]",
            "-map", "[v]", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(compare_mp4),
        ]
        with cmp_log.open("a") as fh:
            fh.write("\n$ " + " ".join(cmd) + "\n")
            fh.flush()
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc != 0 or not compare_mp4.is_file():
            log_line(f"ep{ep:02d} COMPARE_FAIL rc={rc}")
            fail += 1
            continue
        log_line(f"ep{ep:02d} COMPARE_OK {compare_mp4}")
    else:
        log_line(f"ep{ep:02d} COMPARE_SKIP {compare_mp4}")

log_line(f"ALL_DONE fail={fail} out={out}")
sys.exit(1 if fail else 0)
PY
