"""Scale anchoring + workspace frame mapping: V2D world → Libero world.

Two problems the monocular pipeline leaves us with:

1. **Scale** — ViPE/DynHaMR worlds are approximately metric but can drift;
   anchor with a real-world measurement (primary: the object width held during
   the recorded grasp).
2. **Frame** — after V2D's GeoCalib gravity alignment the world is z-up with
   x=right / y=average-camera-forward. The Libero table is z-up with its own
   origin; we map with a translation (origin), a yaw, and optional axis sign
   flips.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .extract import HandTrack


def resolve_scale(
    track: HandTrack,
    method: str = "grasp_aperture",
    known_object_width_m: float | None = None,
    fixed_scale: float = 1.0,
    hold_frame_range: tuple[int, int] | None = None,
) -> float:
    """Return the multiplier converting V2D-world units → metres.

    Methods:
      fixed           — trust the pipeline as-is (`fixed_scale`).
      grasp_aperture  — the hand held a measured-width object during the clip;
                        scale = width / median(aperture in hold window).
                        Window defaults to the frames around the aperture
                        minimum (the grasp moment).
    """
    if method == "fixed":
        return float(fixed_scale)

    if method != "grasp_aperture":
        raise ValueError(f"unknown scale method {method!r}")

    if known_object_width_m is None or known_object_width_m <= 0:
        raise ValueError("grasp_aperture scale needs scale.known_object_width_m")

    ap = track.aperture.copy()
    valid = track.valid & np.isfinite(ap)
    if valid.sum() < 10:
        raise RuntimeError("too few valid aperture samples for scaling")

    if hold_frame_range is None:
        # Grasp moment = global aperture minimum on valid frames; window =
        # ±15% of the clip around it (the hand is holding, not moving toward).
        idx = np.where(valid)[0]
        i_min = idx[np.nanargmin(ap[idx])]
        half = max(5, int(0.15 * len(track.valid)))
        lo, hi = max(0, i_min - half), min(len(track.valid), i_min + half + 1)
        hold = np.zeros_like(valid)
        hold[lo:hi] = True
        hold &= valid
    else:
        lo, hi = hold_frame_range
        hold = np.zeros_like(valid)
        hold[lo:hi] = True
        hold &= valid

    if hold.sum() < 5:
        raise RuntimeError(f"hold window too small ({hold.sum()} valid frames)")

    med = float(np.nanmedian(ap[hold]))
    if med <= 0:
        raise RuntimeError("median hold aperture is not positive")

    scale = known_object_width_m / med
    print(
        f"[calibrate] grasp_aperture: width={known_object_width_m:.3f} m / "
        f"median aperture={med:.4f} → scale={scale:.4f} "
        f"(window frames {lo}..{hi}, n={hold.sum()})"
    )
    return scale


class WorkspaceMap:
    """p_libero = R_z(yaw) @ diag(signs) @ (p_world − origin) × scale."""

    def __init__(
        self,
        origin: np.ndarray | None = None,
        yaw_deg: float = 0.0,
        axis_signs: np.ndarray | None = None,
        scale: float = 1.0,
    ):
        self.origin = np.zeros(3) if origin is None else np.asarray(origin, float)
        self.scale = float(scale)
        signs = np.ones(3) if axis_signs is None else np.asarray(axis_signs, float)
        if not np.all(np.abs(signs) == 1.0):
            raise ValueError("axis_signs must be ±1")
        self.signs = signs
        self.R = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix() @ np.diag(signs)
        det = np.linalg.det(self.R)
        if det < 0:
            print(
                "[calibrate] WARNING: axis_signs make the map a reflection "
                "(det<0). Rotations will be mirrored — prefer fixing "
                "workspace.axis_signs to a proper rotation."
            )

    # --- points -----------------------------------------------------------
    def pos(self, p_world: np.ndarray) -> np.ndarray:
        p = np.asarray(p_world, float)
        return ((p - self.origin) @ self.R.T) * self.scale

    def pos_series(self, P: np.ndarray) -> np.ndarray:
        """(N,3) → (N,3)."""
        return ((np.asarray(P, float) - self.origin) @ self.R.T) * self.scale

    # --- rigid poses ------------------------------------------------------
    def pose(self, T_world: np.ndarray) -> np.ndarray:
        """Map a (4,4) world pose into the Libero frame (scaled translation)."""
        T = np.asarray(T_world, float).copy()
        T[:3, 3] = self.pos(T[:3, 3])
        T[:3, :3] = self.R @ T[:3, :3]
        return T

    def pose_series(self, Ts: np.ndarray) -> np.ndarray:
        out = np.asarray(Ts, float).copy()
        out[:, :3, 3] = self.pos_series(out[:, :3, 3])
        out[:, :3, :3] = np.einsum("ij,njk->nik", self.R, out[:, :3, :3])
        return out

    # --- diagnostics -------------------------------------------------------
    def describe(self) -> str:
        return (
            f"scale={self.scale:.4f} origin={self.origin.tolist()} "
            f"yaw+signs R=\n{np.round(self.R, 4)}"
        )


def default_workspace(cfg: dict, scale: float) -> WorkspaceMap:
    ws = cfg.get("workspace", {}) or {}
    return WorkspaceMap(
        origin=np.asarray(ws.get("origin", [0.0, 0.0, 0.0]), float),
        yaw_deg=float(ws.get("yaw_deg", 0.0)),
        axis_signs=np.asarray(ws.get("axis_signs", [1.0, 1.0, 1.0]), float),
        scale=scale,
    )
