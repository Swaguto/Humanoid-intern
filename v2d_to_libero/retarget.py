"""Retarget: MANO wrist track + aperture → Libero EEF trajectory + gripper.

Steps:
  1. map wrist poses through the calibrated workspace (V2D world → Libero),
  2. apply a fixed wrist-frame → tool-frame offset (rotation + translation),
  3. normalize aperture → continuous gripper command in [0, 1] (1 = open),
  4. resample everything to the control frequency (Libero: 20 Hz).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .calibrate import WorkspaceMap
from .extract import HandTrack


@dataclass
class EEFCommand:
    """Retargeted trajectory in the Libero frame."""

    t: np.ndarray          # (M,) seconds
    pos: np.ndarray        # (M,3) metres
    quat_xyzw: np.ndarray  # (M,4) tool orientation
    gripper: np.ndarray    # (M,) 1 = fully open, 0 = fully closed
    aperture_raw: np.ndarray  # (M,) scaled aperture (m) for diagnostics

    @property
    def n(self) -> int:
        return len(self.t)


def wrist_to_tool_offset(cfg_eef: dict | None) -> np.ndarray:
    """Fixed (4,4) offset applied in the wrist frame: T_tool = T_wrist @ offset."""
    cfg_eef = cfg_eef or {}
    deg = np.asarray(cfg_eef.get("rotation_offset_deg", [0.0, 0.0, 0.0]), float)
    trans = np.asarray(cfg_eef.get("position_offset", [0.0, 0.0, 0.0]), float)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", deg, degrees=True).as_matrix()
    T[:3, 3] = trans
    return T


def aperture_to_gripper(
    ap: np.ndarray,
    valid: np.ndarray,
    open_m: float | str = "auto",
    closed_m: float | str = "auto",
    hysteresis: float = 0.0,
) -> np.ndarray:
    """Normalize scaled aperture (m) → [0,1], 1 = open.

    auto: open = 95th percentile of valid aperture (fully spread reach),
          closed = 20th percentile minus hysteresis margin.
    """
    v = valid & np.isfinite(ap)
    if v.sum() < 5:
        raise RuntimeError("not enough valid aperture samples")
    if open_m == "auto":
        open_m = float(np.percentile(ap[v], 95))
    if closed_m == "auto":
        closed_m = float(np.percentile(ap[v], 20))
    span = float(open_m - closed_m)
    if span <= 1e-6:
        raise RuntimeError(
            f"degenerate aperture range (open={open_m:.4f} closed={closed_m:.4f}) "
            "— set gripper.aperture_open_m/closed_m manually"
        )
    g = (np.asarray(ap, float) - closed_m) / span
    g = np.clip(g, 0.0, 1.0)
    g[~v] = np.nan
    # Forward-fill invalid gaps with the last valid command.
    idx = np.where(np.isfinite(g), np.arange(len(g)), 0)
    np.maximum.accumulate(idx, out=idx)
    g = g[idx]
    g[~np.isfinite(g)] = 1.0
    return g


def _longest_valid_run(track: HandTrack) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the longest consecutive run of valid frames."""
    v = track.valid
    best = (0, 0)
    cur = (0, 0)
    for i, ok in enumerate(v):
        if ok:
            if cur[1] == cur[0] and i > 0 and not v[i - 1]:
                cur = (i, i)
            else:
                cur = (cur[0], i)
            if cur[1] - cur[0] > best[1] - best[0]:
                best = cur
        else:
            cur = (i + 1, i + 1)
    lo, hi = best
    return np.arange(lo, hi + 1), v


def retarget(
    track: HandTrack,
    workspace: WorkspaceMap,
    cfg: dict,
    target_hz: float = 20.0,
) -> EEFCommand:
    """Full retarget of the (longest valid) hand track to EEF commands."""
    idx, _ = _longest_valid_run(track)
    if len(idx) < target_hz:  # < 1 s of data
        raise RuntimeError(f"longest valid run is only {len(idx)} frames")

    # 1-2. wrist poses → Libero frame → tool frame
    T_world = track.wrist_T_world[idx]
    T_lib = workspace.pose_series(T_world)
    T_tool = T_lib @ wrist_to_tool_offset(cfg.get("eef"))

    pos = T_tool[:, :3, 3]
    quat_xyzw = Rotation.from_matrix(T_tool[:, :3, :3]).as_quat()  # xyzw

    # 3. gripper from (scaled) aperture
    ap_m = track.aperture[idx] * workspace.scale
    gcfg = cfg.get("gripper", {}) or {}
    valid = np.ones(len(idx), bool)
    gripper = aperture_to_gripper(
        ap_m,
        valid,
        open_m=gcfg.get("aperture_open_m", "auto"),
        closed_m=gcfg.get("aperture_closed_m", "auto"),
        hysteresis=float(gcfg.get("hysteresis", 0.0)),
    )

    # 4. resample to target_hz (positions linear, orientation slerp, gripper hold)
    src_t = idx / track.fps
    src_t = src_t - src_t[0]
    dst_t = np.arange(0.0, src_t[-1], 1.0 / target_hz)
    if len(dst_t) < 2:
        raise RuntimeError("resampled trajectory too short")

    pos_r = np.stack([np.interp(dst_t, src_t, pos[:, k]) for k in range(3)], axis=1)
    key_times = list(src_t)
    key_rots = Rotation.from_quat(quat_xyzw)
    quat_r = Slerp(key_times, key_rots)(dst_t).as_quat()
    grip_r = np.interp(dst_t, src_t, gripper)
    ap_r = np.interp(dst_t, src_t, ap_m)

    return EEFCommand(t=dst_t, pos=pos_r, quat_xyzw=quat_r, gripper=grip_r, aperture_raw=ap_r)


def summarize_cmd(cmd: EEFCommand) -> str:
    return (
        f"EEFCommand: {cmd.n} steps @ {cmd.n / max(cmd.t[-1], 1e-9):.1f} Hz, "
        f"pos range x[{cmd.pos[:,0].min():.3f},{cmd.pos[:,0].max():.3f}] "
        f"y[{cmd.pos[:,1].min():.3f},{cmd.pos[:,1].max():.3f}] "
        f"z[{cmd.pos[:,2].min():.3f},{cmd.pos[:,2].max():.3f}], "
        f"gripper open-steps={int((cmd.gripper > 0.5).sum())}/{cmd.n}"
    )
