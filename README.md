# From My Hand to a Robot Hand: Turning Phone Video into Libero Demonstrations

Submission for the [Humanoid Internship Challenge](https://jobs.ashbyhq.com/humanoid/e1a2a9de-ad23-4632-9d93-ee50fd41a221)
— *record a phone video of a manipulation task and use it to drive a simulated
manipulator, with your own data playing a real role.*

## TL;DR

Instead of annotating demos by hand or treating real video as "just" domain
randomization, we run the video through **V2D** (NVIDIA's video-to-3D-reconstruction
pipeline), recover the **MANO hand trajectory + aperture**, retarget it into a
**Libero end-effector + gripper action sequence**, replay it in the simulator to
produce **in-domain training demonstrations**, and fine-tune a policy on
*original Libero demos + our video-derived demos*.

```
your phone video
      │
      ▼
┌──────────────┐   wrist SE3 (world), per-frame aperture, object pose
│  V2D ego-e2e │──────────────────────────────────────────────┐
│ (ViPE+DynHaMR│                                              │
│  MANO, depth)│                                              ▼
└──────────────┘                                  ┌──────────────────────┐
                                                  │ v2d_to_libero (new)  │
                                                  │ 1. scale anchor      │
                                                  │ 2. frame → Libero    │
                                                  │ 3. wrist → EEF pose  │
                                                  │ 4. aperture → grip   │
                                                  │ 5. absolute → δ OSC  │
                                                  └──────────┬───────────┘
                                                             ▼
                                                  ┌──────────────────────┐
                                                  │ Libero sim replay    │
                                                  │ (OSC controller,     │
                                                  │  grasp-success       │
                                                  │  filter)             │
                                                  └──────────┬───────────┘
                                                             ▼
                                       in-domain (image, action) demos
                                                             │
                     Libero official demos ──────────────► mix
                                                             ▼
                                            fine-tune π0 / OpenVLA-OFT
                                                             ▼
                                               evaluate on LIBERO suites
```

## Why this is creative (design choices)

1. **Real human motion, not scripted trajectories.** The demo comes from *my*
   hand recorded on a phone — idiosyncratic approach paths, grasp styles, and
   timing that no scripted trajectory generator produces.
2. **V2D does the hard perception for us.** Monocular MANO reconstruction
   (ViPE + Dyn-HaMR), gravity-aligned world frame, object pose tracking — we
   consume its `result.npz` contract rather than reinventing hand tracking.
3. **The replay trick closes the visual gap.** We never feed *real phone frames*
   into the policy. The recovered motion is re-executed in the simulator, so
   the resulting observations are pixel-domain Libero renders — zero
   visual-domain shift in the training data, all the diversity from real human
   kinematics. *(This is the key insight: the value of your video is in the
   **motion distribution**, not the pixels.)*
4. **Aperture is the gripper signal.** Thumb–index fingertip distance from MANO
   FK gives a continuous grasp signal — no grasp labels, no touch sensors.
5. **Physically grounded filtering.** Retargeted trajectories only become
   training demos if the simulator's physics actually succeeds at the task —
   the sim is the labeler.

## Repository layout

```
v2d_to_libero/          the retarget adapter (the new contribution)
  mano.py               pure-numpy MANO forward kinematics (from V2D, Apache-2.0)
  extract.py            result.npz → wrist SE3 track + aperture track
  calibrate.py          scale anchoring + workspace frame → Libero world
  retarget.py           wrist → EEF pose, aperture → gripper, resample @20 Hz
  libero_actions.py     absolute pose → Libero OSC delta actions
scripts/
  run_v2d.sh            wrapper for V2D's run_ego_reconstruction.py
  replay_libero.py      replay retargeted traj, filter success, save demos
  check_conventions.py  verify gripper/rotation conventions before replay
  convert_to_lerobot.py convert successful replay HDF5 to LeRobot
  merge_lerobot.py      combine matching baseline and video datasets
configs/target.yaml     per-experiment configuration
docs/design.md          design decisions & rubric mapping
docs/results.md         evaluation tables & honest failure log
```

## How to run

### 1. Record

Record the **complete task**, including placing and releasing the object in the
target location. For the default configuration, that means moving a black bowl
from the table center onto a plate. Record the true video FPS in
`configs/target.yaml`; keep the object and hand visible from before the grasp
through release. A pickup-only clip cannot pass this task's success check.

### 2. Reconstruct with V2D

Requires a GPU host and the [NVIDIA V2D setup](https://github.com/nvidia-isaac/video_to_data/blob/main/reconstruction/docs/ego_e2e_setup.md).
Use absolute paths because the wrapper runs from V2D's `reconstruction/` directory:

```bash
V2D_ROOT=/path/to/video_to_data ./scripts/run_v2d.sh \
    --video /absolute/path/clip01.mp4 --object_prompt "black bowl" \
    --output_dir /absolute/path/outputs/clip01 --reference_frame 0 \
    --hand_tracking dynhamr --run_droid_slam --run_gravity_alignment
```

### 3. Retarget (CPU only)

```bash
python3 -m v2d_to_libero.extract \
    --result-dir /absolute/path/outputs/clip01/result_slam_gravity_aligned \
    --mano /path/to/mano/models --fps 30 --summary
```

Edit `configs/target.yaml` (result directory, actual FPS, object name, scale,
workspace yaw, and wrist-to-tool offset). The replay uses the V2D object's
initial pose and the simulator object's initial pose to solve translation. It
cannot infer yaw, gripper offset, or metric world scale from that one point.
Inspect a rejected rollout and tune those values before collecting data.
`grasp_aperture` scale anchoring is experimental because MANO hand scale may
not equal the reconstructed world scale.

### 4. Replay in Libero

```bash
python3 scripts/check_conventions.py configs/target.yaml
python3 scripts/replay_libero.py --config configs/target.yaml --save-rejects
# successful episodes: data/demos/*_ok.hdf5
```

### 5. Fine-tune

The repo's HDF5 layout is project-specific; OpenPI's example
`convert_libero_data_to_lerobot.py` reads RLDS and cannot read it directly.
In a LeRobot environment, convert successful episodes with:

```bash
python3 scripts/convert_to_lerobot.py --input-dir data/demos \
    --repo-id local/video_libero --root /path/to/lerobot/local/video_libero
```

The converter downsamples 20 Hz replay to 10 Hz by default, matching the
OpenPI LIBERO example. It exports both camera views, the 8-D state, 7-D
actions, and the task language. For an official baseline already in the same
LeRobot version with an **identical feature schema**, combine datasets with
`python3 scripts/merge_lerobot.py --help`. Then point OpenPI's
`LeRobotLiberoDataConfig.repo_id` at the local baseline or merged dataset and
run the same `pi05_libero` training and evaluation procedure for each. See the
[OpenPI LIBERO guide](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md).
Do not report a policy result until both runs have been evaluated on the same
LIBERO task and initial states.

## Status

| Stage | State |
|---|---|
| V2D pipeline survey + adapter design | done |
| Retarget adapter (`v2d_to_libero/`) | implemented |
| LIBERO replay + demo export | code fixed and unit checked; needs live verification |
| GPU run of V2D on my video | blocked on compute (docker/NVIDIA runtime) |
| LeRobot converter + merge | implemented; needs real-data verification |
| Fine-tune + eval | pending video-derived successful episodes and GPU run |

See `docs/results.md` for numbers as they arrive, including what failed.

## Requirements & data notes

- **MANO models are licensed separately** — put `MANO_RIGHT.pkl` / `MANO_LEFT.pkl`
  in a local dir (e.g. `~/Downloads/mano_v1_2/models/`) and point
  `mano_model_dir` at it. They are never committed.
- `data/`, `*.npz`, `*.hdf5`, `*.mp4` are gitignored; example outputs get
  curated into releases instead.

## Acknowledgments

- [NVIDIA V2D](https://github.com/nvidia-isaac/video_to_data) —
  reconstruction pipeline; `mano.py` adapted from its Apache-2.0 sources.
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — benchmark & sim.
- [openpi](https://github.com/Physical-Intelligence/openpi) / π0,
  [OpenVLA-OFT](https://github.com/siddhanthaldar/OpenVLA-OFT) — policies.

## License

MIT (see `LICENSE`; V2D-derived files keep their Apache-2.0 headers, see
`NOTICE`).
