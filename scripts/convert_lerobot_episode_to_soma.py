#!/usr/bin/env python3
"""Convert one Track 2 LeRobot episode into the soma_to_g1.py folder layout."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _as_array(column, rows: int, width: int) -> np.ndarray:
    values = column.to_pylist()
    out = np.zeros((rows, width), dtype=np.float32)
    for i, item in enumerate(values):
        out[i] = np.asarray(item, dtype=np.float32).reshape(-1)[:width]
    return out


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    n = float(np.linalg.norm(quat))
    if n < 1e-8:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _fill_invalid_poses(poses: np.ndarray, visible: np.ndarray) -> np.ndarray:
    filled = poses.copy()
    last = None
    for i, ok in enumerate(visible):
        if ok:
            last = filled[i]
        elif last is not None:
            filled[i] = last
    last = None
    for i in range(len(visible) - 1, -1, -1):
        if visible[i]:
            last = filled[i]
        elif last is not None:
            filled[i] = last
    return filled


def _joint_names_from_info(info: dict) -> list[str]:
    names = info["features"]["observation.human.pose"]["names"]
    joints: list[str] = []
    for name in names:
        joint = name.rsplit(".", 1)[0]
        if not joints or joints[-1] != joint:
            joints.append(joint)
    return joints


def convert_episode(tier_root: Path, episode_index: int, output_root: Path) -> Path:
    info = json.loads((tier_root / "meta" / "info.json").read_text())
    meta_rows = [
        json.loads(line)
        for line in (tier_root / "meta" / "episodes_metadata.jsonl").read_text().splitlines()
        if line.strip()
    ]
    meta = next(row for row in meta_rows if int(row["episode_index"]) == episode_index)
    sequence_id = meta["sequence_id"]
    parquet_path = tier_root / info["data_path"].format(
        episode_chunk=0, episode_index=episode_index
    )
    table = pq.read_table(parquet_path)
    rows = table.num_rows
    pose = _as_array(table.column("observation.human.pose"), rows, 231).reshape(rows, 77, 3)
    transl = _as_array(table.column("observation.human.translation"), rows, 3)
    identity = _as_array(table.column("observation.human.identity_coeffs"), rows, 45)
    scale = _as_array(table.column("observation.human.scale_params"), rows, 68)
    object_pose = _as_array(table.column("observation.object.pose"), rows, 7).astype(np.float64)
    visible = np.asarray(table.column("observation.object.visible").to_pylist(), dtype=bool)
    object_mats = np.tile(np.eye(4, dtype=np.float64), (rows, 1, 1))
    for i, raw in enumerate(object_pose):
        object_mats[i, :3, :3] = _quat_wxyz_to_matrix(raw[3:7])
        object_mats[i, :3, 3] = raw[:3]
    object_mats = _fill_invalid_poses(object_mats, visible)

    dest = output_root / f"sequence_id={sequence_id}"
    mesh_dir = dest / "reconstructed_mesh"
    dest.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        dest / "soma_params.npz",
        poses=pose,
        transl=transl,
        joint_names=np.asarray(_joint_names_from_info(info)),
        identity_model_type=np.asarray("mhr"),
        identity_coeffs=identity,
        scale_params=scale,
        unit=np.asarray("meters"),
        keep_root=np.asarray(False),
        rotation_repr=np.asarray("rotvec"),
        absolute_pose=np.asarray(False),
    )
    np.save(dest / "poses.npy", object_mats)
    shutil.copy2(tier_root / meta["mesh"], mesh_dir / "output_aligned.glb")
    shutil.copy2(tier_root / meta["ground_plane"], dest / "ground_plane.json")
    support_src = tier_root / meta["support_surface"]
    if support_src.is_file():
        shutil.copy2(support_src, dest / f"{sequence_id}_support.usda")
    (dest / "episode_meta.json").write_text(
        json.dumps(
            {
                "episode_index": episode_index,
                "sequence_id": sequence_id,
                "object": meta["object"],
                "frames": rows,
                "visible_frames": int(visible.sum()),
                "tier_root": str(tier_root),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {dest}  frames={rows} object={meta['object']}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=13)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    convert_episode(args.tier_root.resolve(), args.episode, args.output_root.resolve())


if __name__ == "__main__":
    main()
