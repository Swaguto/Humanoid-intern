import h5py
import numpy as np
import pytest
import sys
import types

from scripts.convert_to_lerobot import convert, validate_episode
from scripts.replay_libero import make_env, rollout, save_episode
from v2d_to_libero.calibrate import WorkspaceMap, align_workspace_to_object
from v2d_to_libero.libero_actions import make_action


def test_object_anchor_maps_video_object_to_sim():
    workspace = WorkspaceMap(yaw_deg=90, scale=2.0)
    video_pos = np.array([1.0, 2.0, 3.0])
    sim_pos = np.array([0.4, -0.2, 0.8])
    align_workspace_to_object(workspace, video_pos, sim_pos)
    np.testing.assert_allclose(workspace.pos(video_pos), sim_pos)


def test_reflection_is_rejected():
    with pytest.raises(ValueError, match="handedness"):
        WorkspaceMap(axis_signs=np.array([-1.0, 1.0, 1.0]))


def test_osc_action_is_normalized():
    action = make_action(
        np.array([0.05, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]),
        np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), 1.0,
        pos_gain=1.0, rot_gain=1.0, pos_clip=0.05, rot_clip=0.5,
    )
    np.testing.assert_allclose(action[:3], [1.0, 0.0, 0.0])


def test_saved_episode_has_training_alignment_and_expected_schema(tmp_path):
    n = 2
    rec = {
        "actions": np.zeros((n, 7)),
        "eef_pos": np.zeros((n, 3)),
        "eef_quat": np.tile([0, 0, 0, 1], (n, 1)),
        "gripper_qpos": np.zeros((n, 2)),
        "state": np.zeros((n, 8)),
        "images": np.zeros((n, 8, 8, 3), np.uint8),
        "wrist_images": np.zeros((n, 8, 8, 3), np.uint8),
        "steps": n,
        "success": True,
    }
    path = tmp_path / "sample_ok.hdf5"
    save_episode(str(path), rec, {"task": "pick and place", "fps": 20})
    with h5py.File(path, "r") as file:
        assert validate_episode(file, path, 8) == n
        assert file["observations/state"].shape == (n, 8)
        assert file.attrs["task"] == "pick and place"


def test_rollout_saves_pre_action_observation():
    class Env:
        def step(self, action):
            next_obs = {
                "robot0_eef_pos": np.array([1.0, 0.0, 0.0]),
                "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "robot0_gripper_qpos": np.zeros(2),
                "agentview_image": np.ones((2, 2, 3), np.uint8),
                "robot0_eye_in_hand_image": np.ones((2, 2, 3), np.uint8),
            }
            return next_obs, 1.0, False, {}

        def check_success(self):
            return True

    obs = {
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
        "agentview_image": np.zeros((2, 2, 3), np.uint8),
        "robot0_eye_in_hand_image": np.zeros((2, 2, 3), np.uint8),
    }
    cmd = type("Command", (), {
        "n": 2, "pos": np.zeros((2, 3)),
        "quat_xyzw": np.tile([0, 0, 0, 1], (2, 1)),
        "gripper": np.ones(2),
    })()
    rec = rollout(Env(), cmd, {"libero": {"camera": "agentview"}}, obs)
    assert rec["success"] and rec["steps"] == 1
    np.testing.assert_allclose(rec["eef_pos"][0], np.zeros(3))
    assert rec["images"][0].sum() == 0


def test_make_env_uses_official_libero_api(monkeypatch):
    calls = {}

    class Suite:
        def get_task_names(self):
            return ["target"]

        def get_task(self, index):
            assert index == 0
            return types.SimpleNamespace(language="pick and place")

        def get_task_bddl_file_path(self, index):
            assert index == 0
            return "/tmp/target.bddl"

        def get_task_init_states(self, index):
            assert index == 0
            return [np.zeros(2)]

    def env_factory(**kwargs):
        calls.update(kwargs)
        return object()

    package = types.ModuleType("libero")
    nested = types.ModuleType("libero.libero")
    nested.benchmark = types.SimpleNamespace(get_benchmark_dict=lambda: {"libero_spatial": Suite})
    envs = types.ModuleType("libero.libero.envs")
    envs.OffScreenRenderEnv = env_factory
    monkeypatch.setitem(sys.modules, "libero", package)
    monkeypatch.setitem(sys.modules, "libero.libero", nested)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", envs)
    _, bddl, language, states = make_env({"libero": {"suite": "libero_spatial", "task": "target"}})
    assert bddl == "/tmp/target.bddl" and language == "pick and place"
    assert len(states) == 1
    assert calls["bddl_file_name"] == bddl
    assert "bddl_file_path" not in calls


def test_converter_downsamples_to_baseline_fps(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    n = 4
    rec = {
        "actions": np.zeros((n, 7)),
        "eef_pos": np.zeros((n, 3)),
        "eef_quat": np.tile([0, 0, 0, 1], (n, 1)),
        "gripper_qpos": np.zeros((n, 2)),
        "state": np.zeros((n, 8)),
        "images": np.zeros((n, 4, 4, 3), np.uint8),
        "wrist_images": np.zeros((n, 4, 4, 3), np.uint8),
        "steps": n, "success": True,
    }
    save_episode(str(inputs / "clip_ok.hdf5"), rec, {"task": "move bowl", "fps": 20})
    frames = []

    class FakeDataset:
        @staticmethod
        def create(**kwargs):
            assert kwargs["fps"] == 10
            return FakeDataset()

        def add_frame(self, frame):
            frames.append(frame)

        def save_episode(self):
            pass

        def finalize(self):
            pass

    package = types.ModuleType("lerobot")
    datasets = types.ModuleType("lerobot.datasets")
    module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    module.LeRobotDataset = FakeDataset
    monkeypatch.setitem(sys.modules, "lerobot", package)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", module)
    assert convert(inputs, "local/test", tmp_path / "out", image_size=4) == 1
    assert len(frames) == 2
    assert frames[0]["task"] == "move bowl"
