from types import SimpleNamespace

import numpy as np

from scripts.video_guided_replay import step


def test_release_continuation_ignores_libero_success_done():
    obs = {
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.02, -0.02]),
        "agentview_image": np.zeros((2, 2, 3), np.uint8),
        "robot0_eye_in_hand_image": np.zeros((2, 2, 3), np.uint8),
    }

    class Env:
        env = SimpleNamespace(done=False)

        def step(self, action):
            return obs, 1.0, True, {}

        def check_success(self):
            return True

    rec = {key: [] for key in (
        "actions", "eef_pos", "eef_quat", "gripper_qpos", "state",
        "images", "wrist_images",
    )}
    action = np.r_[np.zeros(6), -1.0]
    _, stop, success = step(Env(), obs, action, rec, stop_on_success=False)
    assert success and not stop
    assert len(rec["actions"]) == 1

    _, stop, success = step(Env(), obs, action, rec, stop_on_success=True)
    assert success and stop
