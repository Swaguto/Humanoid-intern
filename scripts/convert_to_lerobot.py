#!/usr/bin/env python3
"""Convert successful replay HDF5 episodes to an OpenPI-compatible LeRobot dataset.

Run this in an environment with LeRobot installed. The output uses the keys
expected by OpenPI's LeRobotLiberoDataConfig: image, wrist_image, state, actions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def validate_episode(file: h5py.File, path: Path, image_size: int) -> int:
    if not bool(file.attrs.get("success", False)):
        raise ValueError(f"{path}: only successful episodes may be converted")
    required = ("actions", "observations/state", "observations/images/agentview",
                "observations/images/wrist")
    missing = [name for name in required if name not in file]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    n = len(file["actions"])
    if n == 0 or any(len(file[key]) != n for key in required):
        raise ValueError(f"{path}: empty episode or mismatched sequence lengths")
    if file["actions"].shape[1:] != (7,) or file["observations/state"].shape[1:] != (8,):
        raise ValueError(f"{path}: expected actions (N,7) and state (N,8)")
    for key in required[2:]:
        if file[key].shape[1:] != (image_size, image_size, 3):
            raise ValueError(f"{path}: {key} must be (N,{image_size},{image_size},3)")
    if not np.isfinite(file["actions"][:]).all() or not np.isfinite(file["observations/state"][:]).all():
        raise ValueError(f"{path}: non-finite actions or states")
    return n


def convert(input_dir: Path, repo_id: str, root: Path, image_size: int = 256,
            target_fps: int = 10) -> int:
    paths = sorted(input_dir.glob("*_ok.hdf5"))
    if not paths:
        raise ValueError(f"no successful replay episodes found in {input_dir}")
    if root.exists():
        raise FileExistsError(f"{root} already exists; choose a new --root")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise SystemExit("Install LeRobot in the training environment before converting") from exc

    features = {
        "image": {"dtype": "image", "shape": (image_size, image_size, 3),
                  "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (image_size, image_size, 3),
                        "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }
    with h5py.File(paths[0], "r") as first:
        source_fps = int(round(float(first.attrs["fps"])))
    if source_fps % target_fps:
        raise ValueError(f"source fps {source_fps} must be a multiple of target fps {target_fps}")
    stride = source_fps // target_fps
    dataset = LeRobotDataset.create(repo_id=repo_id, root=root, robot_type="panda",
                                    fps=target_fps, features=features)
    converted = 0
    for path in paths:
        with h5py.File(path, "r") as file:
            n = validate_episode(file, path, image_size)
            if int(round(float(file.attrs["fps"]))) != source_fps:
                raise ValueError(f"{path}: fps differs from first episode ({source_fps})")
            task = str(file.attrs["task"])
            for i in range(0, n, stride):
                dataset.add_frame({
                    "image": file["observations/images/agentview"][i],
                    "wrist_image": file["observations/images/wrist"][i],
                    "state": file["observations/state"][i].astype(np.float32),
                    "actions": file["actions"][i].astype(np.float32),
                    "task": task,
                })
            dataset.save_episode()
            converted += 1
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/demos"))
    parser.add_argument("--repo-id", required=True, help="local dataset ID, e.g. local/video_libero")
    parser.add_argument("--root", type=Path, required=True, help="new local output directory")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--target-fps", type=int, default=10)
    args = parser.parse_args()
    count = convert(args.input_dir, args.repo_id, args.root, args.image_size, args.target_fps)
    print(f"converted {count} successful episodes to {args.root}")


if __name__ == "__main__":
    main()
