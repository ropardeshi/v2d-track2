#!/usr/bin/env python3
"""Train+eval selected Track 2 Tier 1 episodes inside a FlashCHORD container.

Skip an episode when its final policy and eval parquet already exist.
Intended to run via `docker exec` so /home/horde/isaac-automation/v2d_track2
paths in episode_map.json and in the parquets resolve.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MAP_DEFAULT = Path("/home/horde/isaac-automation/v2d_track2/episode_map.json")
ROOT_DEFAULT = Path("/home/horde/isaac-automation/v2d_track2/outputs/tier1_350m")
STAGE = Path("/home/horde/isaac-automation/v2d_track2/retargeted/whole_body/reconstructed_stage")
WORLD_COUNT = 2048
TOTAL_STEPS = 349_999_104
REPLAY_CAPACITY = 10_000_000
REPLAY_MIN = 409_600
SAVE_EVERY = 349_999_104
CURRICULUM = "recon_body_flash_sac"
OBJECT_POSITION_STD = None


def support_enabled(sequence_id: str) -> bool:
    for name in (f"{sequence_id}_g1_support.usda", f"{sequence_id}_support.usda"):
        path = STAGE / name
        if path.is_file():
            text = path.read_text()
            return "Cylinder" in text or "\ndef Cube " in text or "def Cube " in text
    return False


def log_line(log: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        fh.write(line + "\n")


def run_logged(cmd: list[str], logfile: Path) -> int:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a") as fh:
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True, help="comma-separated episode indices")
    parser.add_argument("--map", type=Path, default=MAP_DEFAULT)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--label", default="shard")
    parser.add_argument("--curriculum", default=CURRICULUM)
    parser.add_argument("--object-position-std", type=float, default=OBJECT_POSITION_STD)
    parser.add_argument("--object-orientation-std", type=float, default=None)
    args = parser.parse_args()

    wanted = {int(x) for x in args.episodes.split(",") if x.strip() != ""}
    root: Path = args.root
    log = root / f"train_eval_{args.label}.log"
    ckpt_path = root / f"checkpoints_{args.label}.json"
    episodes = json.loads(args.map.read_text())["episodes"]
    ckpts: dict[str, str] = json.loads(ckpt_path.read_text()) if ckpt_path.exists() else {}

    fail = 0
    seen = 0
    for rec in episodes:
        ep = int(rec["episode_index"])
        if ep not in wanted:
            continue
        seen += 1
        seq = rec["sequence_id"]
        parquet = rec["parquet"]
        ep_dir = root / f"ep{ep:02d}_{seq}"
        train_dir = ep_dir / "train"
        eval_dir = ep_dir / "eval"
        train_log = ep_dir / "train.log"
        eval_log = ep_dir / "eval.log"
        policy = train_dir / f"policy_{TOTAL_STEPS}.safetensors"
        traj = eval_dir / f"episode_{ep:06d}.parquet"
        ep_dir.mkdir(parents=True, exist_ok=True)

        if not Path(parquet).is_file():
            log_line(log, f"ep{ep:02d} MISSING parquet {parquet}")
            fail += 1
            continue

        if not policy.is_file():
            use_support = support_enabled(seq)
            log_line(
                log,
                f"ep{ep:02d} TRAIN {seq} worlds={WORLD_COUNT} steps={TOTAL_STEPS} "
                f"support={use_support} curriculum={args.curriculum}",
            )
            train_dir.mkdir(parents=True, exist_ok=True)
            train_cmd = [
                "python", "-u", "scripts/train_flash_sac.py",
                "experiment=g1_recon_body_flash_sac",
                f"curriculum={args.curriculum}",
                f"task.parquet='{parquet}'",
                "logging.mode=disabled",
                f"scene.world_count={WORLD_COUNT}",
                f"scene.support={str(use_support).lower()}",
                f"training.total_environment_steps={TOTAL_STEPS}",
                f"training.replay.capacity={REPLAY_CAPACITY}",
                f"training.replay.minimum_size={REPLAY_MIN}",
                f"training.checkpoint.save_interval_environment_steps={SAVE_EVERY}",
                f"output_dir={train_dir}",
                f"hydra.run.dir={train_dir}/hydra",
            ]
            if args.object_position_std is not None:
                train_cmd.append(f"objective.object_position_std={args.object_position_std}")
            if args.object_orientation_std is not None:
                train_cmd.append(f"objective.object_orientation_std={args.object_orientation_std}")
            rc = run_logged(train_cmd, train_log)
            for blob in train_dir.glob("state_*.safetensors"):
                blob.unlink()
            if rc != 0 or not policy.is_file():
                log_line(log, f"ep{ep:02d} TRAIN_FAIL rc={rc} policy={policy.is_file()}")
                fail += 1
                continue
            log_line(log, f"ep{ep:02d} TRAIN_OK {policy}")
        else:
            log_line(log, f"ep{ep:02d} TRAIN_SKIP {policy}")

        ckpts[str(ep)] = str(policy)
        ckpt_path.write_text(json.dumps(ckpts, indent=2, sort_keys=True) + "\n")

        if not traj.is_file():
            log_line(log, f"ep{ep:02d} EVAL {seq}")
            eval_dir.mkdir(parents=True, exist_ok=True)
            rc = run_logged(
                [
                    "python", "-u", "scripts/evaluate_policy.py",
                    f"evaluation.checkpoint={policy}",
                    "evaluation.world_count=1",
                    "evaluation.reset_mode=explicit",
                    f"evaluation.metrics_output={eval_dir}/episode_{ep:06d}.json",
                    f"evaluation.object_trajectories_output={traj}",
                    f"hydra.run.dir={eval_dir}/hydra",
                    "logging.mode=disabled",
                ],
                eval_log,
            )
            if rc != 0 or not traj.is_file():
                log_line(log, f"ep{ep:02d} EVAL_FAIL rc={rc} traj={traj.is_file()}")
                fail += 1
                continue
            log_line(log, f"ep{ep:02d} EVAL_OK {traj}")
        else:
            log_line(log, f"ep{ep:02d} EVAL_SKIP {traj}")

    if seen != len(wanted):
        log_line(log, f"WARN wanted={sorted(wanted)} seen={seen}")
        fail += 1
    log_line(log, f"ALL_DONE fail={fail} label={args.label} checkpoints={ckpt_path}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
