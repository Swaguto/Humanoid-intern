import numpy as np

from v2d_to_libero.upright import constrain_bowl_upright


def test_upright_constraint_preserves_position_and_pose_consistency():
    world = np.tile(np.eye(4), (2, 1, 1))
    world[:, :3, 3] = [[0.2, -0.1, 0.9], [0.3, 0.2, 0.9]]
    world[:, :3, :3] = np.diag([1.0, -1.0, -1.0])
    view = np.tile(np.eye(4), (2, 1, 1))
    view[:, :3, 3] = [0.1, 0.0, 0.2]
    camera = np.linalg.inv(view) @ world

    fixed_world, fixed_camera = constrain_bowl_upright(
        world, camera, view, np.array([True, False]), np.ones(2, bool)
    )

    np.testing.assert_allclose(fixed_world[0, :3, :3], np.eye(3))
    np.testing.assert_allclose(fixed_world[:, :3, 3], world[:, :3, 3])
    np.testing.assert_allclose(view[0] @ fixed_camera[0], fixed_world[0])
    np.testing.assert_array_equal(fixed_world[1], world[1])
    np.testing.assert_array_equal(fixed_camera[1], camera[1])
    np.testing.assert_array_equal(world[0, :3, :3], np.diag([1.0, -1.0, -1.0]))
