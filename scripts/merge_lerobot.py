#!/usr/bin/env python3
"""Merge a baseline LeRobot LIBERO dataset with converted video-derived episodes."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo-id", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--video-repo-id", required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"{args.output_root} already exists")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.dataset_tools import merge_datasets
    except ImportError as exc:
        raise SystemExit("Install a recent LeRobot with dataset_tools.merge_datasets") from exc
    baseline = LeRobotDataset(args.baseline_repo_id, root=args.baseline_root)
    video = LeRobotDataset(args.video_repo_id, root=args.video_root)
    if baseline.fps != video.fps:
        raise ValueError(f"FPS mismatch: baseline {baseline.fps}, video {video.fps}")
    if baseline.features != video.features:
        raise ValueError("feature schemas differ; inspect both datasets before merging")
    merged = merge_datasets([baseline, video], output_repo_id=args.output_repo_id,
                            output_dir=args.output_root)
    print(f"merged {merged.meta.total_episodes} episodes into {args.output_root}")


if __name__ == "__main__":
    main()
