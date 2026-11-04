#!/usr/bin/env bash
# Retrain Track 2 Tier 1 with the official G1 FlashSAC recipe, VOC-annealed
# on a 16.8M-step schedule (same stage fractions as 350M). 2048 worlds.
set -u
CONTAINER=flash_chord-latest-gpu0
MAP=/home/horde/isaac-automation/v2d_track2/episode_map.json
ROOT=/home/horde/isaac-automation/v2d_track2/outputs/tier1_scale
WORLD_COUNT=2048
TOTAL_STEPS=16777216
REPLAY_CAPACITY=2097152
REPLAY_MIN=262144
SAVE_EVERY=2097152
LOG=$ROOT/train_eval.log
mkdir -p "$ROOT"
python3 - "$MAP" "$ROOT" "$WORLD_COUNT" "$TOTAL_STEPS" "$REPLAY_CAPACITY" "$REPLAY_MIN" "$SAVE_EVERY" "$CONTAINER" "$LOG" <<'PY'
import json, subprocess, sys, time
from pathlib import Path

(map_path, root, world_s, steps_s, cap_s, min_s, save_s, container, log_path) = sys.argv[1:]
world = int(world_s); steps = int(steps_s); cap = int(cap_s); vmin = int(min_s); save = int(save_s)
root = Path(root)
log = Path(log_path)
episodes = json.loads(Path(map_path).read_text())["episodes"]
ckpts = {}
ckpt_path = root / "checkpoints.json"
if ckpt_path.exists():
    ckpts = json.loads(ckpt_path.read_text())
STAGE = Path("/home/horde/isaac-automation/v2d_track2/retargeted/whole_body/reconstructed_stage")


def support_enabled(sequence_id: str) -> bool:
    for name in (f"{sequence_id}_g1_support.usda", f"{sequence_id}_support.usda"):
        path = STAGE / name
        if path.is_file():
            text = path.read_text()
            return "Cylinder" in text or "\ndef Cube " in text or "def Cube " in text
    return False

def log_line(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with log.open("a") as fh:
        fh.write(line + "\n")

def docker(*args: str, logfile: Path) -> int:
    cmd = ["docker", "exec", "-w", "/workspace", container, "env",
           "PYTHONUNBUFFERED=1", "HYDRA_FULL_ERROR=1", "python", "-u", *args]
    with logfile.open("a") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return proc.returncode

fail = 0
for rec in episodes:
    ep = int(rec["episode_index"])
    seq = rec["sequence_id"]
    parquet = rec["parquet"]
    ep_dir = root / f"ep{ep:02d}_{seq}"
    train_dir = ep_dir / "train"
    eval_dir = ep_dir / "eval"
    train_log = ep_dir / "train.log"
    eval_log = ep_dir / "eval.log"
    policy = train_dir / f"policy_{steps}.safetensors"
    traj = eval_dir / f"episode_{ep:06d}.parquet"
    ep_dir.mkdir(parents=True, exist_ok=True)

    if not Path(parquet).is_file():
        log_line(f"ep{ep:02d} MISSING parquet {parquet}")
        fail += 1
        continue

    if not policy.is_file():
        use_support = support_enabled(seq)
        log_line(f"ep{ep:02d} TRAIN {seq} worlds={world} steps={steps} support={use_support}")
        train_dir.mkdir(parents=True, exist_ok=True)
        rc = docker(
            "scripts/train_flash_sac.py",
            "experiment=g1_recon_body_flash_sac",
            "curriculum=recon_body_flash_sac_fast16m",
            f"task.parquet='{parquet}'",
            "logging.mode=disabled",
            f"scene.world_count={world}",
            f"scene.support={str(use_support).lower()}",
            f"training.total_environment_steps={steps}",
            f"training.replay.capacity={cap}",
            f"training.replay.minimum_size={vmin}",
            f"training.checkpoint.save_interval_environment_steps={save}",
            f"output_dir={train_dir}",
            f"hydra.run.dir={train_dir}/hydra",
            logfile=train_log,
        )
        if rc != 0 or not policy.is_file():
            log_line(f"ep{ep:02d} TRAIN_FAIL rc={rc} policy={policy.is_file()}")
            fail += 1
            continue
        log_line(f"ep{ep:02d} TRAIN_OK {policy}")
    else:
        log_line(f"ep{ep:02d} TRAIN_SKIP {policy}")

    ckpts[str(ep)] = str(policy)
    ckpt_path.write_text(json.dumps(ckpts, indent=2, sort_keys=True) + "\n")

    if not traj.is_file():
        log_line(f"ep{ep:02d} EVAL {seq}")
        eval_dir.mkdir(parents=True, exist_ok=True)
        rc = docker(
            "scripts/evaluate_policy.py",
            f"evaluation.checkpoint={policy}",
            "evaluation.world_count=1",
            "evaluation.reset_mode=explicit",
            f"evaluation.metrics_output={eval_dir}/episode_{ep:06d}.json",
            f"evaluation.object_trajectories_output={traj}",
            f"hydra.run.dir={eval_dir}/hydra",
            "logging.mode=disabled",
            logfile=eval_log,
        )
        if rc != 0 or not traj.is_file():
            log_line(f"ep{ep:02d} EVAL_FAIL rc={rc} traj={traj.is_file()}")
            fail += 1
            continue
        log_line(f"ep{ep:02d} EVAL_OK {traj}")
    else:
        log_line(f"ep{ep:02d} EVAL_SKIP {traj}")

log_line(f"ALL_DONE fail={fail} checkpoints={ckpt_path}")
sys.exit(1 if fail else 0)
PY
