#!/usr/bin/env python3
"""Replay a phone-video hand track in LIBERO and save successful episodes."""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2d_to_libero.calibrate import align_workspace_to_object, default_workspace, resolve_scale
from v2d_to_libero.extract import load_hand_track, load_object_track
from v2d_to_libero.libero_actions import build_env_action
from v2d_to_libero.retarget import retarget, summarize_cmd


def make_env(cfg: dict):
    """Return env, BDDL path, task language, and official initial states."""
    try:
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as exc:
        raise SystemExit("Install LIBERO first: pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git") from exc

    lcfg = cfg["libero"]
    suite_name = lcfg.get("suite", "libero_spatial")
    suites = benchmark.get_benchmark_dict()
    if suite_name not in suites:
        raise ValueError(f"unknown LIBERO suite {suite_name!r}; choices: {sorted(suites)}")
    suite = suites[suite_name]()
    names = suite.get_task_names()
    task_name = lcfg["task"]
    if task_name not in names:
        raise ValueError(f"task {task_name!r} not in {suite_name}; choices: {names}")
    task_id = names.index(task_name)
    task = suite.get_task(task_id)
    bddl = lcfg.get("bddl_file_path") or suite.get_task_bddl_file_path(task_id)
    init_states = suite.get_task_init_states(task_id)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=int(lcfg.get("camera_size", 256)),
        camera_widths=int(lcfg.get("camera_size", 256)),
        control_freq=int(lcfg.get("control_freq", 20)),
    )
    return env, bddl, task.language, init_states


def sim_object_position(env, cfg: dict) -> np.ndarray:
    """Read the target object's actual position after LIBERO initial-state reset."""
    ocfg = cfg.get("object", {})
    if ocfg.get("sim_position") is not None:
        return np.asarray(ocfg["sim_position"], float).reshape(3)
    name = ocfg.get("libero_name")
    if not name:
        raise ValueError("set object.libero_name (LIBERO object name) or object.sim_position")
    ids = env.env.obj_body_id
    if name not in ids:
        raise ValueError(f"object {name!r} unavailable; LIBERO object names: {sorted(ids)}")
    return np.asarray(env.env.sim.data.body_xpos[ids[name]], float).copy()


def video_object_position(object_track) -> np.ndarray:
    valid = np.flatnonzero(object_track.valid)
    if not len(valid):
        raise ValueError("V2D has no valid object pose frames; cannot align the workspace")
    first = valid[: min(len(valid), 10)]
    return np.median(object_track.pose_T_world[first, :3, 3], axis=0)


def reset_episode(env, init_state, seed: int):
    env.seed(seed)
    obs = env.reset()
    if init_state is not None:
        obs = env.set_init_state(init_state)
    return obs


def rollout(env, cmd, cfg: dict, obs: dict) -> dict:
    """Collect each observation before its action, matching training alignment."""
    camera = cfg["libero"].get("camera", "agentview")
    rec = {key: [] for key in ("actions", "eef_pos", "eef_quat", "gripper_qpos",
                               "images", "wrist_images", "state")}
    success = False
    for i in range(cmd.n):
        action = build_env_action(cmd.pos[i], cmd.quat_xyzw[i], obs,
                                  float(cmd.gripper[i]), cfg)
        pos = np.asarray(obs["robot0_eef_pos"], np.float32)
        quat = np.asarray(obs["robot0_eef_quat"], np.float64)
        grip = np.asarray(obs["robot0_gripper_qpos"], np.float32).reshape(-1)
        if grip.size != 2:
            raise ValueError(f"expected 2 gripper joint positions, got {grip.shape}")
        state = np.concatenate((pos, Rotation.from_quat(quat).as_rotvec(), grip)).astype(np.float32)
        rec["actions"].append(action)
        rec["eef_pos"].append(pos)
        rec["eef_quat"].append(quat)
        rec["gripper_qpos"].append(grip)
        rec["state"].append(state)
        rec["images"].append(np.asarray(obs[f"{camera}_image"], np.uint8))
        rec["wrist_images"].append(np.asarray(obs["robot0_eye_in_hand_image"], np.uint8))
        obs, _, done, _ = env.step(action)
        success = success or bool(env.check_success())
        if done or success:
            break
    for key in rec:
        rec[key] = np.asarray(rec[key])
    rec["success"] = success
    rec["steps"] = len(rec["actions"])
    return rec


def save_episode(path: str, rec: dict, meta: dict) -> None:
    """Save a project-specific episode consumed by convert_to_lerobot.py."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("actions", data=rec["actions"].astype(np.float32))
        obs = f.create_group("observations")
        images = obs.create_group("images")
        images.create_dataset("agentview", data=rec["images"], compression="gzip")
        images.create_dataset("wrist", data=rec["wrist_images"], compression="gzip")
        for name in ("eef_pos", "eef_quat", "gripper_qpos", "state"):
            obs.create_dataset(name, data=rec[name])
        f.attrs["success"] = rec["success"]
        f.attrs["num_samples"] = rec["steps"]
        for key, value in meta.items():
            f.attrs[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--save-rejects", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    result_dir = args.result_dir or cfg["v2d_result_dir"]
    track = load_hand_track(result_dir, side=cfg.get("hand", "right"),
                            mano_model_dir=os.path.expanduser(cfg["mano_model_dir"]),
                            fps_override=cfg.get("video_fps"))
    object_track = load_object_track(result_dir)
    scfg = cfg.get("scale", {})
    scale = resolve_scale(track, method=scfg.get("method", "fixed"),
                          known_object_width_m=scfg.get("known_object_width_m"),
                          fixed_scale=float(scfg.get("fixed_scale", 1.0)),
                          hold_frame_range=tuple(scfg["hold_frame_range"]) if "hold_frame_range" in scfg else None)
    env, bddl, language, init_states = make_env(cfg)
    out_dir = args.out or cfg.get("output_dir", "data/demos")
    os.makedirs(out_dir, exist_ok=True)
    n_ok = 0
    try:
        for ep in range(args.episodes):
            init_state = init_states[ep % len(init_states)] if len(init_states) else None
            obs = reset_episode(env, init_state, int(cfg["libero"].get("seed", 0)) + ep)
            workspace = default_workspace(cfg, scale)
            align_workspace_to_object(workspace, video_object_position(object_track),
                                      sim_object_position(env, cfg))
            cmd = retarget(track, workspace, cfg,
                           target_hz=float(cfg["libero"].get("control_freq", 20)))
            print(f"episode {ep}: {workspace.describe()}\n{summarize_cmd(cmd)}")
            rec = rollout(env, cmd, cfg, obs)
            n_ok += int(rec["success"])
            tag = "ok" if rec["success"] else "fail"
            print(f"episode {ep}: {tag}, {rec['steps']} steps")
            if rec["success"] or args.save_rejects:
                filename = f"{os.path.basename(os.path.dirname(result_dir.rstrip('/')))}_{ep}_{tag}.hdf5"
                save_episode(os.path.join(out_dir, filename), rec,
                             {"source": "v2d_phone_video", "bddl": bddl,
                              "task": language, "scale": scale, "fps": float(cfg["libero"].get("control_freq", 20))})
    finally:
        env.close()
    print(f"done: {n_ok}/{args.episodes} successful -> {out_dir}")


if __name__ == "__main__":
    main()
