"""Constrain a bowl proxy's opening to world up after V2D gravity alignment.

This is an explicit postprocessing assumption, not a recovered V2D rotation.
The proxy mesh is rotationally symmetric, so its yaw is set to zero.
"""

from __future__ import annotations

import numpy as np


def constrain_bowl_upright(
    object_to_world: np.ndarray,
    object_to_camera: np.ndarray,
    camera_to_world: np.ndarray,
    object_valid: np.ndarray,
    camera_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return consistent world/camera poses with the bowl's local +Z upright.

    Object translations and all invalid frames are preserved. The input world
    frame must be gravity aligned with +Z up, and the mesh must open along +Z.
    """
    world = np.asarray(object_to_world)
    camera = np.asarray(object_to_camera)
    view = np.asarray(camera_to_world)
    valid = np.asarray(object_valid, dtype=bool) & np.asarray(camera_valid, dtype=bool)
    if world.ndim != 3 or world.shape[1:] != (4, 4) or camera.shape != world.shape or view.shape != world.shape:
        raise ValueError("expected matching (frames, 4, 4) pose arrays")
    if valid.shape != (len(world),):
        raise ValueError("validity arrays must have one entry per frame")
    if not np.isfinite(world[valid]).all() or not np.isfinite(view[valid]).all():
        raise ValueError("valid poses must be finite")

    corrected_world = world.copy()
    corrected_camera = camera.copy()
    corrected_world[valid, :3, :3] = np.eye(3, dtype=world.dtype)
    corrected_camera[valid] = np.linalg.inv(view[valid]) @ corrected_world[valid]
    return corrected_world, corrected_camera
