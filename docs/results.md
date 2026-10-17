# Results

## Video-guided LIBERO replay

`IMG_7805.MOV` is an applicant-recorded bowl-to-plate demonstration. Thirteen
bowl-center waypoints were read from frames resized to 512×288; the arm hides the
bowl in some frames, so this is a sparse inspected track. The replay uses the
first five official LIBERO demonstrations for grasp and release and replaces
their transport phase with video-driven end-effector targets.

| Evaluation | Attempts | Successes | Mean steps | Criterion |
|---|---:|---:|---:|---|
| Video-guided transport, `libero_spatial` bowl-to-plate | 5 | 5 | 197.4 | `env.check_success()` |

The five successful HDF5 episodes have 7-D actions, 8-D robot states, and both
camera views aligned before each action. Their lengths are 190, 206, 201, 191,
and 199 steps. [Machine-readable metrics](video_guided_metrics.json) and an
[example final render](media/video_guided_demo_0_final.png) are included.
The original full-resolution MOV and licensed MANO models are not committed.
An authorized [short source-video excerpt](media/IMG_7805_excerpt.mp4) shows
the recorded task.

The official `demo_0` action sequence also passed a direct simulator replay
(1/1), confirming that this installation of LIBERO and robosuite can execute
the reference grasp. This is a controlled replay result, not a trained policy
evaluation. It does not measure generalization to new objects or tasks.

Environment: LIBERO `8f1084e`, robosuite 1.4.0, MuJoCo 2.3.7, NumPy 1.26.4,
Python 3.10, headless EGL on an NVIDIA L40S. The task is
`pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`.

## Full video-to-hand retargeting

The V2D/MANO reconstruction and full hand-retargeted LIBERO rollout have not
yet produced a verified result. No claim about their success rate or policy
fine-tuning is made here.

## Failures and corrections

- A scripted grasp at the bowl center missed. The successful expert grasp
  approaches near the rim with a lateral offset; the video-guided baseline
  retains that expert grasp.
- LIBERO's stored task state initially placed the bowl at about 0.97 m, while
  it settled to about 0.898 m on the table. `replay_libero.py` now takes open-
  gripper settling steps before measuring the simulator bowl for calibration.
- The applicant's real bowl has a handle, unlike the simulator's bowl. The
  video-guided replay transfers the transport trajectory, not the real grasp
  geometry. V2D currently uses an approximate bowl mesh, which may affect
  object-pose quality.
