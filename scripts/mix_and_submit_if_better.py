#!/usr/bin/env python3
"""Rebuild the T1 mix from every leftover eval and upload if it beats hardmix."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/horde/isaac-automation/v2d_track2/outputs")
PACK = ROOT / "tier1_freeze_pack"
CSV = ROOT / "track2_tier1_freeze.csv"
SOURCES_OUT = ROOT / "tier1_freeze_sources.json"
KIT = Path("/home/horde/isaac-automation/v2d_submission_kit/v2d_submission_kit")
PY = Path("/home/horde/isaac-automation/v2d_track2/.venv/bin/python")
COMMIT = "https://github.com/nvidia-isaac/video_to_data/commit/0593d80de5d39ab6f56f73ed16ca464ba5053499"
LIVE = (0.43394, 0.53333, 0.30000, 13.04419)  # AUC, SP, MP, MPPE

RUNS = {
    "hardmix": ROOT / "tier1_hardmix_pack",
    "short": ROOT / "tier1",
    "16.8M": ROOT / "tier1_scale",
    "350M": ROOT / "tier1_350m",
    "hard": ROOT / "tier1_hard",
    "orient": ROOT / "tier1_orient",
}


def load_run(root: Path, tag: str) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if not root.exists():
        return rows
    for path in root.rglob("episode_*.json"):
        if "hydra" in path.parts:
            continue
        parquet = path.with_suffix(".parquet")
        if not parquet.is_file():
            continue
        try:
            report = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        metrics = report.get("metrics") or {}
        try:
            episode = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        pos = float(metrics["object_position_error_m"])
        ori_deg = float(metrics["object_orientation_error_deg"])
        ori_rad = float(metrics.get("object_orientation_error_rad") or math.radians(ori_deg))
        sp = 1.0 if pos <= 0.10 and ori_rad <= 0.5 else 0.0
        mp = 1.0 if pos <= 0.03 and ori_deg <= 30.0 else 0.0
        row = {
            "tag": tag,
            "auc": float(metrics.get("add_auc") or 0.0),
            "sp": sp,
            "mp": mp,
            "mppe": float(metrics.get("mppe_cm") or metrics.get("mean_per_point_error_cm") or 1e9),
            "json": path,
            "parquet": parquet,
        }
        prev = rows.get(episode)
        if prev is None or _key(row) > _key(prev):
            rows[episode] = row
    return rows


def _key(row: dict) -> tuple:
    return (row["sp"], row["mp"], -row["mppe"], row["auc"])


def mix() -> tuple[dict[int, dict], dict[str, str]]:
    pool: dict[int, list[dict]] = {ep: [] for ep in range(30)}
    for tag, root in RUNS.items():
        for ep, row in load_run(root, tag).items():
            pool[ep].append(row)
    chosen: dict[int, dict] = {}
    sources: dict[str, str] = {}
    missing = []
    for ep in range(30):
        if not pool[ep]:
            missing.append(ep)
            continue
        best = max(pool[ep], key=_key)
        chosen[ep] = best
        sources[str(ep)] = best["tag"]
    if missing:
        raise SystemExit(f"missing episodes in mix: {missing}")
    return chosen, sources


def stage(chosen: dict[int, dict]) -> None:
    if PACK.exists():
        shutil.rmtree(PACK)
    PACK.mkdir(parents=True)
    for ep, row in chosen.items():
        shutil.copy2(row["parquet"], PACK / f"episode_{ep:06d}.parquet")
        shutil.copy2(row["json"], PACK / f"episode_{ep:06d}.json")


def scores(chosen: dict[int, dict]) -> tuple[float, float, float, float]:
    n = len(chosen)
    auc = sum(r["auc"] for r in chosen.values()) / n
    sp = sum(r["sp"] for r in chosen.values()) / n
    mp = sum(r["mp"] for r in chosen.values()) / n
    mppe = sum(r["mppe"] for r in chosen.values()) / n
    return auc, sp, mp, mppe


def better(new: tuple[float, float, float, float], old: tuple[float, float, float, float]) -> bool:
    # Rank-sum cares about SP/MP/AUC up and MPPE down. Never ship a mix that
    # drops SP or MP versus the live hardmix.
    n_auc, n_sp, n_mp, n_mppe = new
    o_auc, o_sp, o_mp, o_mppe = old
    if n_sp + 1e-9 < o_sp or n_mp + 1e-9 < o_mp:
        return False
    return (n_sp, n_mp, n_auc, -n_mppe) > (o_sp, o_mp, o_auc, -o_mppe)


def pack() -> None:
    cmd = [
        str(PY),
        str(KIT / "tools/pack_submission.py"),
        "--track",
        "track_2_tier1",
        "--results",
        str(PACK),
        "--dataset-root",
        "/home/horde/isaac-automation/video_to_data_challenge",
        "--code-commit",
        COMMIT,
        "-o",
        str(CSV),
    ]
    subprocess.check_call(cmd, cwd=str(KIT))


def submit() -> None:
    receipt = ROOT / "track2_tier1_freeze.kaggle-receipt.json"
    cmd = [
        str(PY),
        str(KIT / "submit/v2d_submit.py"),
        str(CSV),
        "--track",
        "track_2_tier1",
        "-m",
        "Tier 1 freeze mix: best-of plus orientation near-miss retrain",
        "--wait",
        "--receipt",
        str(receipt),
    ]
    subprocess.check_call(cmd, cwd=str(KIT))


def main() -> int:
    chosen, sources = mix()
    auc, sp, mp, mppe = scores(chosen)
    print(
        f"mix AUC={auc:.5f} SP={sp:.5f} MP={mp:.5f} MPPE={mppe:.4f} "
        f"sources={json.dumps(sources, sort_keys=True)}"
    )
    SOURCES_OUT.write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")
    if not better((auc, sp, mp, mppe), LIVE):
        print("no improvement over live hardmix; skip pack/upload")
        return 0
    stage(chosen)
    pack()
    if "--no-submit" in sys.argv:
        print(f"packed {CSV}; submit skipped")
        return 0
    submit()
    print("uploaded freeze mix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
