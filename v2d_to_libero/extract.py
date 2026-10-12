"""Extract hand/object tracks from a V2D ego-e2e ``result/`` bundle.

Consumes the contract written by ``v2d.common.result_bundle.write_result_bundle``:

  result.npz
    camera_to_world_transform       (N,4,4)
    camera_intrinsics               [fx,fy,cx,cy]
    object_to_world_transform       (N,4,4), object_scale, object_is_valid
    hand_{left,right}_wrist_to_world_transform  (N,4,4)
    hand_{left,right}_is_valid                  (N,)
    hand_{left,right}_betas                     (10,)
    hand_{left,right}_wrist_orient_in_camera    (3, per frame)
    hand_{left,right}_wrist_trans_in_camera     (3, per frame)
    hand_{left,right}_finger_pose               (45, per frame)   raw axis-angle
    hand_{left,right}_scale                     (N,)
    gravity_alignment_applied / gravity_alignment_transform  (if GeoCalib ran)
  manifest.json
    fps, gravity_alignment{enabled,target,...}, world_coordinate_convention
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import numpy as np

from .mano import ManoModel, aperture, load_mano_model, mano_fk


@dataclass
class HandTrack:
    """Per-frame wrist pose in the V2D world frame + grip aperture."""

    side: str
    fps: float
    frames: np.ndarray            # (M,) source frame indices (valid frames only)
    valid: np.ndarray             # (N,) original per-frame validity
    wrist_T_world: np.ndarray     # (N,4,4) — NaN rows where invalid
    aperture: np.ndarray          # (N,) thumb–index distance, raw V2D units
    scale: np.ndarray             # (N,) per-frame DynHaMR hand scale
    gravity_aligned: bool = False
    world_convention: dict = field(default_factory=dict)

    @property
    def valid_frames(self) -> np.ndarray:
        return self.frames

    @property
    def valid_wrist_T(self) -> np.ndarray:
        """(M,4,4) wrist poses for valid frames only."""
        return self.wrist_T_world[self.frames]

    @property
    def valid_aperture(self) -> np.ndarray:
        return self.aperture[self.frames]


@dataclass
class ObjectTrack:
    frames: np.ndarray            # (N,) all frames
    pose_T_world: np.ndarray      # (N,4,4)
    scale: float
    valid: np.ndarray             # (N,)


def _load_manifest(result_dir: str) -> dict:
    path = os.path.join(result_dir, "manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_hand_track(
    result_dir: str,
    side: str = "right",
    mano_model_dir: str | None = None,
    mano_model: ManoModel | None = None,
    mano_pkl: str | None = None,
    fps_override: float | None = None,
    require_gravity_aligned: bool = True,
) -> HandTrack:
    """Load wrist track + compute per-frame aperture via MANO FK.

    The aperture is computed in the camera frame — a rigid transform does not
    change fingertip distances, so world alignment does not matter for it.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be left|right, got {side!r}")

    manifest = _load_manifest(result_dir)
    grav = manifest.get("gravity_alignment", {})
    gravity_aligned = bool(grav.get("enabled")) and grav.get("target") == "z_up"
    if require_gravity_aligned and not gravity_aligned:
        raise RuntimeError(
            f"{result_dir}/result is NOT gravity-aligned to z_up. "
            "Re-run V2D with GeoCalib gravity alignment (result bundle "
            "manifest.gravity_alignment.enabled == true) or pass "
            "require_gravity_aligned=False to override."
        )

    npz_path = os.path.join(result_dir, "result.npz")
    with np.load(npz_path) as z:
        T_world = z[f"hand_{side}_wrist_to_world_transform"].astype(np.float64)  # (N,4,4)
        valid = z[f"hand_{side}_is_valid"].astype(bool)                          # (N,)
        betas = z[f"hand_{side}_betas"].astype(np.float64)                       # (10,)
        wrist_orient = z[f"hand_{side}_wrist_orient_in_camera"].astype(np.float64)
        wrist_trans = z[f"hand_{side}_wrist_trans_in_camera"].astype(np.float64)
        finger_pose = z[f"hand_{side}_finger_pose"].astype(np.float64)
        hand_scale = z[f"hand_{side}_scale"].astype(np.float64)                  # (N,)

    if mano_model is None:
        if mano_pkl is None:
            if mano_model_dir is None:
                raise ValueError("provide mano_pkl or mano_model_dir")
            name = "MANO_RIGHT.pkl" if side == "right" else "MANO_LEFT.pkl"
            mano_pkl = os.path.join(mano_model_dir, name)
        mano_model = load_mano_model(mano_pkl)

    n = T_world.shape[0]
    ap = np.full(n, np.nan)
    for i in range(n):
        if not valid[i]:
            continue
        verts, _ = mano_fk(
            mano_model,
            betas=betas,
            global_orient=wrist_orient[i],
            transl=wrist_trans[i],
            pose45=finger_pose[i],
        )
        ap[i] = aperture(verts, scale=float(hand_scale[i]))

    # Frames where FK exploded → treat as invalid.
    finite = np.isfinite(ap)
    valid = valid & finite

    fps = fps_override or float(manifest.get("fps", 30.0))
    frames = np.nonzero(valid)[0]

    return HandTrack(
        side=side,
        fps=fps,
        frames=frames,
        valid=valid,
        wrist_T_world=T_world,
        aperture=ap,
        scale=hand_scale,
        gravity_aligned=gravity_aligned,
        world_convention=manifest.get("world_coordinate_convention", {}),
    )


def load_object_track(result_dir: str) -> ObjectTrack:
    npz_path = os.path.join(result_dir, "result.npz")
    with np.load(npz_path) as z:
        pose = z["object_to_world_transform"].astype(np.float64)
        valid = z["object_is_valid"].astype(bool)
        scale = float(z["object_scale"])
    n = pose.shape[0]
    return ObjectTrack(
        frames=np.arange(n),
        pose_T_world=pose,
        scale=scale,
        valid=valid,
    )


def summarize(track: HandTrack) -> str:
    ap = track.valid_aperture
    lines = [
        f"side={track.side}  fps={track.fps}  frames={len(track.frames)}/{len(track.valid)}"
        f"  gravity_aligned={track.gravity_aligned}",
        f"aperture (raw units): min={np.nanmin(ap):.4f} "
        f"p10={np.nanpercentile(ap, 10):.4f} median={np.nanmedian(ap):.4f} "
        f"p90={np.nanpercentile(ap, 90):.4f} max={np.nanmax(ap):.4f}",
    ]
    pos = track.valid_wrist_T[:, :3, 3]
    lines.append(
        "wrist world range: "
        + ", ".join(
            f"{ax}=[{pos[:, k].min():.3f},{pos[:, k].max():.3f}]"
            for k, ax in enumerate("xyz")
        )
    )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result-dir", required=True, help="V2D output .../result directory")
    p.add_argument("--side", default="right", choices=["left", "right"])
    p.add_argument("--mano", required=True, help="dir containing MANO_RIGHT.pkl")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--allow-unaligned", action="store_true")
    p.add_argument("--summary", action="store_true")
    p.add_argument("--npz-out", default=None, help="write cleaned track npz here")
    args = p.parse_args()

    track = load_hand_track(
        args.result_dir,
        side=args.side,
        mano_model_dir=args.mano,
        fps_override=args.fps,
        require_gravity_aligned=not args.allow_unaligned,
    )
    if args.summary:
        print(summarize(track))
    if args.npz_out:
        np.savez_compressed(
            args.npz_out,
            frames=track.frames,
            wrist_T_world=track.wrist_T_world,
            aperture=track.aperture,
            valid=track.valid,
            fps=track.fps,
        )
        print(f"wrote {args.npz_out}")


if __name__ == "__main__":
    main()
