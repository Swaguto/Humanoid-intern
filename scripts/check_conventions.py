#!/usr/bin/env python3
"""Verify environment conventions before trusting any replay.

Libero forks differ in two conventions that silently corrupt demos if wrong:

1. gripper sign — which command value opens the fingers;
2. rotation delta frame — whether OSC rotation deltas are world- or
   end-effector-frame (we assume world; a mismatch shows up as axis-swirl).

Run once per environment install:

    python scripts/check_conventions.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_gripper(env) -> str:
    obs = env.reset()
    q_open0 = float(np.asarray(obs["robot0_gripper_qpos"]).max())

    for value, label in [(+1.0, "+1"), (-1.0, "-1")]:
        obs = env.reset()
        for _ in range(30):
            a = np.zeros(7)
            a[-1] = value
            obs, *_ = env.step(a)
        q = float(np.asarray(obs["robot0_gripper_qpos"]).max())
        print(f"  gripper cmd {label} over 30 steps → finger qpos {q_open0:.3f} → {q:.3f}")
        if q > q_open0 + 0.01:
            print(f"  → {label} OPENS the gripper")
        elif q < q_open0 - 0.01:
            print(f"  → {label} CLOSES the gripper")

    print("  set libero.gripper_convention in configs/target.yaml to match:")
    print("    cmd −1 opens, +1 closes → 'neg1_open'  (robosuite default)")
    print("    cmd +1 opens, −1 closes → 'pos1_open'")
    return "done"


def check_rotation(env) -> None:
    obs = env.reset()
    q0 = np.asarray(obs["robot0_eef_quat"], float)
    print(f"  start eef quat (xyzw): {np.round(q0, 3)}")
    for _ in range(30):
        a = np.zeros(7)
        a[3] = 0.2   # droll
        obs, *_ = env.step(a)
    q1 = np.asarray(obs["robot0_eef_quat"], float)
    print(f"  after +0.2 droll ×30 → quat {np.round(q1, 3)}")
    print("  (expected: roll changed, yaw/pitch ≈ unchanged. If axes are")
    print("   permuted, rotation deltas are NOT world-frame — adjust rot frame)")


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo_root, "scripts"))
    from replay_libero import make_env
    import yaml

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/target.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env, bddl, _, init_states = make_env(cfg)
    if len(init_states):
        env.reset()
        env.set_init_state(init_states[0])
    print(f"env ready: {bddl}")
    print("\n[1] gripper convention")
    check_gripper(env)
    print("\n[2] rotation delta frame")
    check_rotation(env)
    env.close()


if __name__ == "__main__":
    main()
