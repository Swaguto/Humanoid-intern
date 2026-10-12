"""Libero OSC action construction: absolute EEF target → 7-dim delta action.

Libero (robosuite OSC_POSE) action layout:  [dx, dy, dz, droll, dpitch, dyaw, g]
where translation/rotation are *deltas* and the gripper is absolute.

The per-step delta is a proportional correction toward the next waypoint:

    dpos  = clip(pos_gain * (target_pos − current_pos))
    drot  = clip(rot_gain * axis_angle(R_current → R_target))   (world frame)
    g     = gripper command in the env's convention

Conventions (gripper sign, rotation frame) must be verified once per
environment build with scripts/check_conventions.py — Libero forks differ.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def make_action(
    target_pos: np.ndarray,
    target_quat_xyzw: np.ndarray,
    cur_pos: np.ndarray,
    cur_quat_xyzw: np.ndarray,
    gripper_unit: float,
    *,
    pos_gain: float,
    rot_gain: float,
    pos_clip: float,
    rot_clip: float,
) -> np.ndarray:
    """One 7-dim Libero action driving current pose → target pose."""
    dpos = pos_gain * (np.asarray(target_pos, float) - np.asarray(cur_pos, float))
    dpos = np.clip(dpos, -pos_clip, pos_clip)

    R_cur = Rotation.from_quat(np.asarray(cur_quat_xyzw, float))
    R_tgt = Rotation.from_quat(np.asarray(target_quat_xyzw, float))
    R_err = R_tgt * R_cur.inv()
    drot = rot_gain * R_err.as_rotvec()          # world-frame axis-angle delta
    drot = np.clip(drot, -rot_clip, rot_clip)

    return np.concatenate([dpos, drot, [gripper_unit]]).astype(np.float64)


def gripper_to_env(gripper_unit: float, convention: str) -> float:
    """Map our 1=open/0=closed to the env's gripper command.

    conventions:
      neg1_open   — −1 open, +1 close (robosuite default for several grippers)
      pos1_open   — +1 open, −1 close
      zero_one    — 1 open, 0 close (some Libero dataset conventions)
    """
    g = float(np.clip(gripper_unit, 0.0, 1.0))
    if convention == "neg1_open":
        return 1.0 - 2.0 * g          # g=1(open) → −1, g=0(closed) → +1
    if convention == "pos1_open":
        return 2.0 * g - 1.0          # g=1(open) → +1, g=0(closed) → −1
    if convention == "zero_one":
        return g
    raise ValueError(f"unknown gripper convention {convention!r}")


def build_env_action(
    target_pos: np.ndarray,
    target_quat_xyzw: np.ndarray,
    obs: dict,
    gripper_unit: float,
    cfg: dict,
) -> np.ndarray:
    """Convenience: pull current pose from a Libero obs dict, build the action."""
    rcfg = cfg.get("replay", {})
    cur_pos = np.asarray(obs["robot0_eef_pos"], float)
    cur_quat = np.asarray(obs["robot0_eef_quat"], float)  # xyzw
    action = make_action(
        target_pos,
        target_quat_xyzw,
        cur_pos,
        cur_quat,
        gripper_unit,
        pos_gain=float(rcfg.get("pos_gain", 1.0)),
        rot_gain=float(rcfg.get("rot_gain", 1.0)),
        pos_clip=float(rcfg.get("pos_clip", 0.05)),
        rot_clip=float(rcfg.get("rot_clip", 0.3)),
    )
    action[-1] = gripper_to_env(
        gripper_unit, cfg.get("libero", {}).get("gripper_convention", "neg1_open")
    )
    return action
