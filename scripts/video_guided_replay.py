#!/usr/bin/env python3
"""Use measured phone-video bowl motion for the transport phase of a LIBERO demo.

The official demonstration supplies the grasp and other non-transport actions.
The replay may stop at first physics success before release. During transport, the
robot follows positions and timing derived from the applicant's video rather
than replaying the official transport actions. This is a deliberately limited
baseline alongside the full V2D retargeting experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2d_to_libero.libero_actions import build_env_action


def load_waypoints(path: Path, start_s: float, end_s: float) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    points = np.array([[float(row[k]) for k in ("time_s", "bowl_cx_px", "bowl_cy_px")]
                       for row in rows], dtype=float)
    if len(points) < 2 or np.any(np.diff(points[:, 0]) <= 0):
        raise ValueError("waypoint times must be strictly increasing")
    points = points[(points[:, 0] >= start_s) & (points[:, 0] <= end_s)]
    if len(points) < 2 or points[0, 0] != start_s or points[-1, 0] != end_s:
        raise ValueError("start/end times must have annotated waypoints")
    if abs(points[0, 1] - points[-1, 1]) < 1:
        raise ValueError("bowl horizontal displacement is too small")
    return points


def load_v2d_object_points(path: Path, start_s: float, end_s: float) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("V2D object CSV needs at least two rows")
    points = np.array([[float(row[k]) for k in ("time_s", "x_m", "y_m", "z_m")]
                       for row in rows], dtype=float)
    if len(points) < 2 or not np.isfinite(points).all() or np.any(np.diff(points[:, 0]) <= 0):
        raise ValueError("V2D object points must be finite with increasing times")
    points = points[(points[:, 0] >= start_s - 1e-6) & (points[:, 0] <= end_s + 1e-6)]
    if len(points) < 2 or abs(points[0, 0] - start_s) > 1e-6 or abs(points[-1, 0] - end_s) > 1e-6:
        raise ValueError("start/end times must be present in the V2D object track")
    if abs(points[0, 1] - points[-1, 1]) < 0.01:
        raise ValueError("V2D object x displacement is too small")
    return points


def phase_indices(actions: np.ndarray) -> tuple[int, int]:
    grip = actions[:, -1]
    close = np.flatnonzero((grip[1:] > 0) & (grip[:-1] < 0))
    if not len(close):
        raise ValueError("expert demo has no gripper-close transition")
    release = np.flatnonzero((grip[1:] < 0) & (grip[:-1] > 0))
    release = release[release > close[0]]
    if not len(release):
        raise ValueError("expert demo has no gripper-release transition")
    start = int(close[0] + 25)
    end = int(release[0] - 18)
    if end <= start:
        raise ValueError("expert transport phase is too short")
    return start, end


def record_step(rec: dict, obs: dict, action: np.ndarray) -> None:
    pos = np.asarray(obs["robot0_eef_pos"], np.float32)
    quat = np.asarray(obs["robot0_eef_quat"], np.float32)
    grip = np.asarray(obs["robot0_gripper_qpos"], np.float32).reshape(-1)
    if grip.size != 2:
        raise ValueError(f"expected 2 gripper joints, got {grip.size}")
    rec["actions"].append(np.asarray(action, np.float32))
    rec["eef_pos"].append(pos)
    rec["eef_quat"].append(quat)
    rec["gripper_qpos"].append(grip)
    rec["state"].append(np.r_[pos, Rotation.from_quat(quat).as_rotvec(), grip].astype(np.float32))
    rec["images"].append(np.asarray(obs["agentview_image"], np.uint8))
    rec["wrist_images"].append(np.asarray(obs["robot0_eye_in_hand_image"], np.uint8))


def step(env, obs: dict, action: np.ndarray, rec: dict) -> tuple[dict, bool]:
    record_step(rec, obs, action)
    obs, _, done, _ = env.step(action)
    return obs, bool(done or env.check_success())


def run_demo(env, demo, cfg: dict, points: np.ndarray, fps: int,
             mode: str = "video") -> tuple[dict, dict, dict]:
    actions = np.asarray(demo["actions"])
    start, end = phase_indices(actions)
    pos = np.asarray(demo["obs/ee_pos"])
    rotvec = np.asarray(demo["obs/ee_ori"])
    orient = Slerp([0.0, 1.0], Rotation.from_rotvec(rotvec[[start, end]]))
    rec = {k: [] for k in ("actions", "eef_pos", "eef_quat", "gripper_qpos",
                           "state", "images", "wrist_images")}
    env.reset()
    obs = env.set_init_state(np.asarray(demo.attrs["init_state"]))
    if mode == "expert":
        for action in actions:
            obs, done = step(env, obs, action, rec)
            if done:
                break
    else:
        for action in actions[:start]:
            obs, done = step(env, obs, action, rec)
            if done:
                break

        if not done:
            t0, t1 = points[0, 0], points[-1, 0]
            x0, x1 = points[0, 1], points[-1, 1]
            y0 = points[0, 2]
            for t in np.linspace(t0, t1, round((t1 - t0) * fps) + 1):
                x = np.interp(t, points[:, 0], points[:, 1])
                y = np.interp(t, points[:, 0], points[:, 2])
                progress = float(np.clip((x0 - x) / (x0 - x1), 0.0, 1.0))
                target = pos[start] * (1 - progress) + pos[end] * progress
                if mode == "v2d":
                    z = np.interp(t, points[:, 0], points[:, 3])
                    baseline_z = points[0, 3] * (1 - progress) + points[-1, 3] * progress
                    lift = max(0.0, z - baseline_z)
                else:
                    lift = 0.0015 * max(0.0, y0 - y)
                target[2] = max(pos[start, 2], pos[end, 2]) + lift
                quat = orient([progress]).as_quat()[0]
                action = build_env_action(target, quat, obs, 0.0, cfg)
                obs, done = step(env, obs, action, rec)
                if done:
                    break

        if not done:
            for action in actions[end:]:
                obs, done = step(env, obs, action, rec)
                if done:
                    break

    success = bool(env.check_success())
    result = {"expert_demo": demo.name.rsplit("/", 1)[-1],
              "success": success, "steps": len(rec["actions"]),
              "expert_transport_start": start, "expert_transport_end": end,
              "final_bowl_pos": np.asarray(obs["akita_black_bowl_1_pos"]).round(4).tolist()}
    return rec, result, obs


def save_episode(path: Path, rec: dict, result: dict, task: str, fps: int,
                 mode: str) -> None:
    with h5py.File(path, "w") as f:
        for key in rec:
            data = np.asarray(rec[key])
            if key in ("images", "wrist_images"):
                group = f.require_group("observations/images")
                group.create_dataset("agentview" if key == "images" else "wrist",
                                     data=data, compression="gzip")
            elif key == "actions":
                f.create_dataset(key, data=data)
            else:
                f.require_group("observations").create_dataset(key, data=data)
        f.attrs["source"] = {"video": "applicant_video_guided_transport",
                              "v2d": "applicant_v2d_object_guided_transport",
                              "expert": "libero_expert_replay"}[mode]
        f.attrs["task"] = task
        f.attrs["success"] = result["success"]
        f.attrs["num_samples"] = result["steps"]
        f.attrs["fps"] = float(fps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/target.yaml")
    parser.add_argument("--expert-hdf5", required=True)
    parser.add_argument("--waypoints", default="data/derived/bowl_to_plate_waypoints.csv")
    parser.add_argument("--v2d-object-csv", default="data/derived/bowl_to_plate_v2d_object_track.csv")
    parser.add_argument("--start-s", type=float, default=9.0)
    parser.add_argument("--end-s", type=float, default=15.0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out-dir", default="data/video_guided_replay")
    parser.add_argument("--save-failures", action="store_true")
    parser.add_argument("--mode", choices=("video", "v2d", "expert"), default="video")
    args = parser.parse_args()

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    points = (load_v2d_object_points(Path(args.v2d_object_csv), args.start_s, args.end_s)
              if args.mode == "v2d" else load_waypoints(Path(args.waypoints), args.start_s, args.end_s))
    suite = benchmark.get_benchmark_dict()[cfg["libero"]["suite"]]()
    task = cfg["libero"]["task"]
    task_id = suite.get_task_names().index(task)
    env = OffScreenRenderEnv(
        bddl_file_name=suite.get_task_bddl_file_path(task_id),
        camera_heights=int(cfg["libero"].get("camera_size", 128)),
        camera_widths=int(cfg["libero"].get("camera_size", 128)),
        control_freq=int(cfg["libero"].get("control_freq", 20)),
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        with h5py.File(args.expert_hdf5) as demos:
            for i in range(args.episodes):
                demo = demos[f"data/demo_{i}"]
                rec, result, obs = run_demo(env, demo, cfg, points,
                                            int(cfg["libero"].get("control_freq", 20)),
                                            args.mode)
                results.append(result)
                print(result, flush=True)
                if result["success"] or args.save_failures:
                    tag = "ok" if result["success"] else "fail"
                    save_episode(out / f"{args.mode}_demo_{i}_{tag}.hdf5", rec,
                                 result, task, int(cfg["libero"].get("control_freq", 20)),
                                 args.mode)
                Image.fromarray(np.flip(obs["agentview_image"], axis=0)).save(
                    out / f"{args.mode}_demo_{i}_final.png")
    finally:
        env.close()
    summary = {"task": task, "mode": args.mode,
               "video_waypoints": str(args.waypoints) if args.mode == "video" else None,
               "v2d_object_track": str(args.v2d_object_csv) if args.mode == "v2d" else None,
               "video_transport_s": [args.start_s, args.end_s],
               "attempts": len(results), "successes": sum(r["success"] for r in results),
               "episodes": results}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"success: {summary['successes']}/{summary['attempts']}", flush=True)


if __name__ == "__main__":
    main()
