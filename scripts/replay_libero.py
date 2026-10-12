#!/usr/bin/env python3
"""Replay a retargeted V2D trajectory in Libero; save grasp-successful demos.

Pipeline stage: retargeted (EEF pose + gripper) → Libero OSC control loop →
physics rollout → success filter → HDF5 demo (observations + actions).

Each saved episode is an *in-domain* Libero demonstration: images are Libero
renders, actions are the deltas the controller received — the phone video
contributed the motion, the simulator guarantees physical plausibility.

Usage:
    python scripts/replay_libero.py --config configs/target.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import h5py
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2d_to_libero.calibrate import default_workspace, resolve_scale
from v2d_to_libero.extract import load_hand_track
from v2d_to_libero.libero_actions import build_env_action
from v2d_to_libero.retarget import retarget


def make_env(cfg: dict):
    """Build a Libero OffScreenRenderEnv from config."""
    try:
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "libero is not installed. Install with:\n"
            "  pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git\n"
            f"(original error: {e})"
        )

    lcfg = cfg["libero"]
    bddl = lcfg.get("bddl_file_path")
    if not bddl:
        suite_name = lcfg.get("suite", "libero_10")
        suite = benchmark.get_libero_suite() if suite_name == "libero_suite" else None
        if suite is None:
            getter = getattr(benchmark, f"get_{suite_name}", None)
            if getter is None:
                raise SystemExit(
                    f"unknown libero suite {suite_name!r}; set libero.bddl_file_path"
                )
            suite = getter()
        task = suite.get_task(lcfg["task"])
        bddl = task.bddl_file_path

    env = OffScreenRenderEnv(
        bddl_file_path=bddl,
        camera_heights=int(lcfg.get("camera_size", 256)),
        camera_widths=int(lcfg.get("camera_size", 256)),
    )
    return env, bddl


def rollout(env, cmd, cfg: dict) -> dict:
    """Execute one retargeted trajectory; return rollout record."""
    obs = env.reset()
    records = {"actions": [], "eef_pos": [], "eef_quat": [], "gripper_qpos": [],
               "images": []}
    camera = cfg["libero"].get("camera", "agentview")
    seed = int(cfg["libero"].get("seed", 0))
    np.random.seed(seed)

    for i in range(cmd.n):
        action = build_env_action(cmd.pos[i], cmd.quat_xyzw[i], obs,
                                  float(cmd.gripper[i]), cfg)
        obs, reward, done, info = env.step(action)

        records["actions"].append(action)
        records["eef_pos"].append(np.asarray(obs["robot0_eef_pos"], np.float32))
        records["eef_quat"].append(np.asarray(obs["robot0_eef_quat"], np.float32))
        records["gripper_qpos"].append(np.asarray(obs["robot0_gripper_qpos"], np.float32))
        records["images"].append(np.asarray(obs[f"{camera}_image"], np.uint8))
        if done:
            break

    success = bool(env.check_success())
    for k in records:
        records[k] = np.asarray(records[k])
    records["success"] = success
    records["steps"] = len(records["actions"])
    return records


def save_episode(path: str, rec: dict, meta: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("actions", data=rec["actions"].astype(np.float32))
        img_grp = f.create_group("observations/images")
        img_grp.create_dataset("agentview", data=rec["images"], compression="gzip")
        obs = f.create_group("observations")
        obs.create_dataset("eef_pos", data=rec["eef_pos"])
        obs.create_dataset("eef_quat", data=rec["eef_quat"])
        obs.create_dataset("gripper_qpos", data=rec["gripper_qpos"])
        f.attrs["success"] = rec["success"]
        for k, v in meta.items():
            f.attrs[k] = v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--result-dir", default=None, help="override v2d result dir")
    ap.add_argument("--episodes", type=int, default=1,
                    help="rollout attempts (perturb start slightly each try)")
    ap.add_argument("--save-rejects", action="store_true",
                    help="also save failed rollouts (for debugging)")
    ap.add_argument("--out", default=None, help="output dir override")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    result_dir = args.result_dir or cfg["v2d_result_dir"]
    mano_dir = os.path.expanduser(cfg["mano_model_dir"])
    side = cfg.get("hand", "right")

    print(f"[1/3] loading V2D track from {result_dir}")
    track = load_hand_track(result_dir, side=side, mano_model_dir=mano_dir)

    print("[2/3] calibrating + retargeting")
    scfg = cfg.get("scale", {})
    scale = resolve_scale(
        track,
        method=scfg.get("method", "grasp_aperture"),
        known_object_width_m=scfg.get("known_object_width_m"),
        fixed_scale=float(scfg.get("fixed_scale", 1.0)),
        hold_frame_range=tuple(scfg["hold_frame_range"]) if "hold_frame_range" in scfg else None,
    )
    workspace = default_workspace(cfg, scale)
    print(workspace.describe())
    cmd = retarget(track, workspace, cfg, target_hz=float(cfg["libero"].get("control_freq", 20)))
    from v2d_to_libero.retarget import summarize_cmd
    print(summarize_cmd(cmd))

    print("[3/3] building Libero env + replaying")
    env, bddl = make_env(cfg)
    print(f"  task bddl: {bddl}")

    out_dir = args.out or cfg.get("output_dir", "data/demos")
    os.makedirs(out_dir, exist_ok=True)
    n_ok = 0
    for ep in range(args.episodes):
        rec = rollout(env, cmd, cfg)
        tag = "ok" if rec["success"] else "fail"
        print(f"  episode {ep}: {tag}  steps={rec['steps']}")
        keep = rec["success"] or args.save_rejects
        if keep:
            name = f"{os.path.splitext(os.path.basename(result_dir.rstrip('/')))[0]}_{ep}_{tag}.hdf5"
            save_episode(
                os.path.join(out_dir, name), rec,
                meta={"source": "v2d_phone_video", "bddl": bddl,
                      "scale": scale, "gains": str(cfg.get("replay", {}))},
            )
        n_ok += int(rec["success"])

    env.close()
    print(f"done: {n_ok}/{args.episodes} successful → {out_dir}")


if __name__ == "__main__":
    main()
