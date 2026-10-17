# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for this project from the NVIDIA V2D Challenge repository
# (reconstruction/modules/v2d_hand_alignment/lib/recover_mano_params.py).
# Pure-numpy MANO forward kinematics — no torch/manotorch/chumpy required.
"""MANO forward kinematics (numpy only).

Loads the (separately licensed) MANO ``.pkl`` models, runs shape + pose
blend-skinning forward kinematics, and exposes fingertip vertices for the
aperture signal used by the gripper retargeter.
"""

from __future__ import annotations

import pickle
import sys
import types
from dataclasses import dataclass

import numpy as np

# Fingertip vertex ids on the MANO mesh (topology shared by MANO_RIGHT/MANO_LEFT).
# Source: V2D scripts/track3/refinement_report.py :: MANO_TIP_VERTS.
TIP_VERTS = {"thumb": 745, "index": 317, "middle": 444, "ring": 556, "little": 673}


def rodrigues(r: np.ndarray) -> np.ndarray:
    """(3,) axis-angle → (3,3) rotation matrix."""
    theta = float(np.linalg.norm(r))
    if theta < 1e-8:
        return np.eye(3)
    n = r / theta
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _install_chumpy_stub() -> None:
    """MANO pickles reference ``chumpy`` objects; stub them just enough to load."""
    if "chumpy" in sys.modules:
        return

    class _Ch:
        def __new__(cls, *args, **kwargs):
            return object.__new__(cls)

        def __init__(self, x=None, *args, **kwargs):
            self._state = {"x": np.asarray(x)} if x is not None else {}

        def __setstate__(self, state):
            self._state = state if isinstance(state, dict) else {"x": state}

        def _resolve(self) -> np.ndarray:
            state = self._state
            if "x" in state:
                v = state["x"]
                return v._resolve() if isinstance(v, _Ch) else np.asarray(v)
            if "a" in state and "idxs" in state:
                a = state["a"]
                a = a._resolve() if isinstance(a, _Ch) else np.asarray(a)
                idxs = np.asarray(state["idxs"])
                result = a.flatten()[idxs]
                ps = state.get("preferred_shape")
                return result.reshape(ps) if ps is not None else result
            return np.array([])

        def __array__(self, dtype=None):
            return np.asarray(self._resolve(), dtype=dtype)

        @property
        def r(self):
            return self._resolve()

    class _ChumMod(types.ModuleType):
        def __getattr__(self, name):
            # Import and traceback tools probe module metadata such as
            # __file__. Returning a Ch class for those breaks later imports
            # (notably torch via LIBERO's benchmark module).
            if name.startswith("__"):
                raise AttributeError(name)
            return _Ch

    stub = _ChumMod("chumpy")
    stub.__path__ = []  # type: ignore[attr-defined]
    stub.Ch = _Ch  # type: ignore[attr-defined]
    sys.modules["chumpy"] = stub
    for _sub in ["reordering", "utils", "ch", "logic"]:
        _m = _ChumMod(f"chumpy.{_sub}")
        sys.modules[f"chumpy.{_sub}"] = _m
        setattr(stub, _sub, _m)


@dataclass
class ManoModel:
    v_template: np.ndarray      # (778, 3)
    shapedirs: np.ndarray       # (778, 3, 10)
    posedirs: np.ndarray        # (778, 3, 45)
    weights: np.ndarray         # (778, 16)
    hands_mean: np.ndarray      # (45,)
    parents: np.ndarray         # (16,)
    J_regressor: np.ndarray     # (16, 778)


def load_mano_model(pkl_path: str) -> ManoModel:
    """Load a MANO ``.pkl`` into plain numpy arrays."""
    _install_chumpy_stub()
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    return ManoModel(
        v_template=np.array(raw["v_template"], dtype=np.float64),
        shapedirs=np.array(raw["shapedirs"], dtype=np.float64),
        posedirs=np.array(raw["posedirs"], dtype=np.float64),
        weights=np.array(raw["weights"], dtype=np.float64),
        hands_mean=np.array(raw["hands_mean"], dtype=np.float64),
        parents=np.array(raw["kintree_table"][0], dtype=np.int32),
        J_regressor=np.asarray(raw["J_regressor"].todense(), dtype=np.float64),
    )


def mano_fk(
    model: ManoModel,
    betas: np.ndarray,
    pose45: np.ndarray,
    global_orient: np.ndarray | None = None,
    transl: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """MANO forward kinematics.

    Args:
        betas:    (10,) shape coefficients.
        pose45:   (45,) finger pose, raw axis-angle WITHOUT ``hands_mean``
                  (DynHaMR/V2D convention — see recover_mano_params.py).
        global_orient: (3,) axis-angle root orientation (camera frame).
        transl:   (3,) root translation (camera frame).

    Returns:
        verts:  (778, 3) mesh vertices.
        joints: (16, 3) joint positions (0 = wrist).
    """
    betas = np.asarray(betas, dtype=np.float64).reshape(-1)
    pose45 = np.asarray(pose45, dtype=np.float64).reshape(-1)
    root = np.zeros(3) if global_orient is None else np.asarray(global_orient, dtype=np.float64).reshape(3)
    t = np.zeros(3) if transl is None else np.asarray(transl, dtype=np.float64).reshape(3)

    full_pose = np.concatenate([root, pose45])                      # (48,)
    v_shaped = model.v_template + np.einsum("ijk,k->ij", model.shapedirs, betas)
    # NOTE: some numpy 2.0.x + BLAS builds (macOS) emit spurious
    # divide-by-zero/overflow/invalid RuntimeWarnings for *any* matmul even
    # with finite inputs — results verified correct — so wrap the heavy FP ops.
    with np.errstate(all="ignore"):
        J = model.J_regressor @ v_shaped                             # (16, 3)
        R = np.stack([rodrigues(full_pose[3 * i : 3 * i + 3]) for i in range(16)])

        pose_feature = (R[1:] - np.eye(3)).reshape(-1)               # (45,)
        v_posed = v_shaped + np.einsum("ijk,k->ij", model.posedirs, pose_feature)

    parents = model.parents
    G = np.zeros((16, 4, 4))
    for k in range(16):
        local = np.eye(4)
        local[:3, :3] = R[k]
        local[:3, 3] = J[k] if k == 0 else J[k] - J[parents[k]]
        G[k] = local if k == 0 else G[parents[k]] @ local

    # G already includes the root rotation (R[0] enters through G[0]); the
    # V2D/HaMeR convention is simply FK(...) + transl — see
    # render_mano_params_video.py line 160: return (v_out + transl).
    joints = G[:, :3, 3] + t

    G_final = np.zeros((16, 4, 4))
    for k in range(16):
        offset = np.eye(4)
        offset[:3, 3] = -J[k]
        G_final[k] = G[k] @ offset
    with np.errstate(all="ignore"):
        T = np.einsum("vk,kij->vij", model.weights, G_final)
        v_homo = np.concatenate([v_posed, np.ones((len(v_posed), 1))], axis=1)
        v_out = np.einsum("vij,vj->vi", T, v_homo)[:, :3] + t

    return v_out.astype(np.float64), joints.astype(np.float64)


def fingertip_positions(verts: np.ndarray) -> dict[str, np.ndarray]:
    """Named fingertip positions from mesh vertices: {finger: (3,)}."""
    return {name: verts[idx].copy() for name, idx in TIP_VERTS.items()}


def aperture(verts: np.ndarray, scale: float = 1.0) -> float:
    """Thumb–index fingertip distance (metres after `scale`)."""
    tips = fingertip_positions(verts)
    return float(np.linalg.norm(tips["thumb"] - tips["index"]) * scale)
