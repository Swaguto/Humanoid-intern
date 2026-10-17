#!/usr/bin/env python3
"""Export only the tracked object positions from a private V2D result bundle."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--start-s", type=float, default=9.0)
    parser.add_argument("--end-s", type=float, default=15.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.fps <= 0 or args.end_s <= args.start_s:
        raise ValueError("fps must be positive and end-s must exceed start-s")
    with np.load(args.result_dir / "result.npz") as bundle:
        pose = bundle["object_to_world_transform"]
        valid = bundle["object_is_valid"].astype(bool)
    if pose.ndim != 3 or pose.shape[1:] != (4, 4) or valid.shape != (len(pose),):
        raise ValueError("unexpected V2D object track dimensions")
    indices = np.arange(len(pose))
    selected = indices[(indices / args.fps >= args.start_s - 1e-6)
                       & (indices / args.fps <= args.end_s + 1e-6)]
    if len(selected) < 2 or not valid[selected].all():
        raise ValueError("object poses must be valid throughout the selected interval")
    xyz = pose[selected, :3, 3]
    if not np.isfinite(xyz).all():
        raise ValueError("object poses contain non-finite coordinates")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", "x_m", "y_m", "z_m"))
        for frame, (x, y, z) in zip(selected, xyz):
            writer.writerow((f"{frame / args.fps:.1f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"))
    print(f"wrote {len(selected)} object poses to {args.out}")


if __name__ == "__main__":
    main()
