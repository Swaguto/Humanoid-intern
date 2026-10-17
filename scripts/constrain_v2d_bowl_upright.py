#!/usr/bin/env python3
"""Write a separate V2D bundle with an explicitly gravity-upright bowl proxy."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v2d_to_libero.upright import constrain_bowl_upright


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True,
                        help="Original gravity-aligned V2D bundle directory")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="New directory; the original bundle is never modified")
    args = parser.parse_args()
    source = args.result_dir.resolve()
    output = args.out_dir.resolve()
    if source == output or output.exists():
        raise ValueError("out-dir must be a new directory separate from result-dir")

    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("world_coordinate_convention", {}).get("z_axis") != "up":
        raise ValueError("input bundle must use a gravity-aligned +Z-up world")
    if manifest.get("mesh", {}).get("mesh") != "mesh.obj":
        raise ValueError("expected the bowl proxy as mesh.obj")

    with np.load(source / "result.npz", allow_pickle=False) as saved:
        bundle = {key: saved[key] for key in saved.files}
    if not bool(bundle["gravity_alignment_applied"]):
        raise ValueError("input bundle must have gravity alignment applied")
    world, camera = constrain_bowl_upright(
        bundle["object_to_world_transform"],
        bundle["object_to_camera_transform"],
        bundle["camera_to_world_transform"],
        bundle["object_is_valid"],
        bundle["camera_is_valid"],
    )
    bundle["object_to_world_transform"] = world
    bundle["object_to_camera_transform"] = camera
    manifest["postprocessing"] = {
        "type": "gravity_upright_bowl_proxy",
        "description": "Object rotation replaced with world identity; translations and hand poses preserved.",
        "raw_v2d_orientation_recovered": False,
    }

    output.mkdir(parents=True)
    np.savez_compressed(output / "result.npz", **bundle)
    shutil.copy2(source / "mesh.obj", output / "mesh.obj")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote gravity-upright bowl bundle to {output}")


if __name__ == "__main__":
    main()
