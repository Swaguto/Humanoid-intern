# From My Hand to a Robot Hand: Turning Phone Video into Libero Demonstrations

Submission for the [Humanoid Internship Challenge](https://jobs.ashbyhq.com/humanoid/e1a2a9de-ad23-4632-9d93-ee50fd41a221)
— *record a phone video of a manipulation task and use it to drive a simulated
manipulator, with your own data playing a real role.*

## TL;DR

Instead of annoting demos by hand or treating real video as "just" domain
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
  run_v2d.sh            wrapper for V2D's run_v2d_ego_e2e.py
  replay_libero.py      replay retargeted traj, filter success, save demos
  check_conventions.py  verify gripper/rotation conventions before replay
configs/target.yaml     per-experiment configuration
docs/design.md          design decisions & rubric mapping
docs/results.md         evaluation tables & honest failure log
```

## How to run

### 1. Record

Short clip (5–15 s, 30 fps, ≥720p) of picking up an object that exists in the
target Libero task. Measure the object's width — you need it for scale.

### 2. Reconstruct with V2D

Requires a GPU host (docker + `--runtime=nvidia`, or a host venv — see
[V2D repo](https://github.com/Swaguto/NVIDIA-V2D-Challenge)):

```bash
./scripts/run_v2d.sh --video_path data/raw/clip01.mp4 --prompt "red mug" \
    --output_dir data/outputs/clip01 --depth_source vipe
```

### 3. Retarget (CPU only)

```bash
python -m v2d_to_libero.extract --result-dir data/outputs/clip01/result \
    --mano ~/Downloads/mano_v1_2/models --summary
# → prints aperture series + trajectory stats, writes retarget config hints
```

Edit `configs/target.yaml` (object width, workspace origin, Libero task).

### 4. Replay in Libero

```bash
python scripts/check_conventions.py      # one-time: verify gripper/rot conventions
python scripts/replay_libero.py --config configs/target.yaml
# → data/demos/clip01.hdf5 (only grasp-successful rollouts are kept)
```

### 5. Fine-tune

Mix `data/demos/*.hdf5` with the official Libero demos and fine-tune with
[openpi](https://github.com/Physical-Intelligence/openpi) (`pi05_libero`-style
config) or [OpenVLA-OFT](https://github.com/siddhanthaldar/OpenVLA-OFT).
Convert with `convert_libero_data_to_lerobot.py`, then evaluate on the LIBERO
suites. **Report baseline (official demos only) vs +phone-video demos.**

## Status

| Stage | State |
|---|---|
| V2D pipeline survey + adapter design | done |
| Retarget adapter (`v2d_to_libero/`) | implemented |
| Libero replay + demo export | implemented, needs live verification |
| GPU run of V2D on my video | blocked on compute (docker/NVIDIA runtime) |
| Fine-tune + eval | pending |

See `docs/results.md` for numbers as they arrive, including what failed.

## Requirements & data notes

- **MANO models are licensed separately** — put `MANO_RIGHT.pkl` / `MANO_LEFT.pkl`
  in a local dir (e.g. `~/Downloads/mano_v1_2/models/`) and point
  `mano_model_dir` at it. They are never committed.
- `data/`, `*.npz`, `*.hdf5`, `*.mp4` are gitignored; example outputs get
  curated into releases instead.

## Acknowledgments

- [NVIDIA V2D Challenge](https://github.com/Swaguto/NVIDIA-V2D-Challenge) —
  reconstruction pipeline; `mano.py` adapted from its Apache-2.0 sources.
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) — benchmark & sim.
- [openpi](https://github.com/Physical-Intelligence/openpi) / π0,
  [OpenVLA-OFT](https://github.com/siddhanthaldar/OpenVLA-OFT) — policies.

## License

MIT (see `LICENSE`; V2D-derived files keep their Apache-2.0 headers, see
`NOTICE`).
