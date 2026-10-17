# Results

## Video-guided LIBERO replay

`bowl_to_plate_demonstration.mov` is my recorded bowl-to-plate demonstration.
Thirteen bowl-center waypoints were read from frames resized to 512×288; the arm hides the
bowl in some frames, so this is a sparse inspected track. The replay uses the
first five official LIBERO demonstrations for grasp and non-transport actions
and replaces their transport phase with video-driven end-effector targets.
The original five-run evaluations end at first physics success before gripper
release. A separate extended run below continues through release.

| Evaluation | Attempts | Successes | Mean steps | Criterion |
|---|---:|---:|---:|---|
| Video-guided transport, `libero_spatial` bowl-to-plate | 5 | 5 | 197.4 | `env.check_success()` |
| V2D 3D object-guided transport, same initial states | 5 | 5 | 197.8 | `env.check_success()` |
| Official expert actions replayed on the same five initial states | 5 | 5 | 98.0 | `env.check_success()` |

The five successful HDF5 episodes have 7-D actions, 8-D robot states, and both
camera views aligned before each action. Their lengths are 190, 206, 201, 191,
and 199 steps. [Machine-readable metrics](video_guided_metrics.json) and an
[example final render](media/video_guided_demo_0_final.png) are included.
The V2D object-guided episodes have the same schema and lengths 190, 206, 202,
190, and 201. All five passed the converter's episode validation; see their
[metrics](v2d_object_metrics.json) and [example final render](media/v2d_object_demo_0_final.png).
The [full two-camera simulator replay](media/full_libero_bowl_placement.mp4)
shows an extended V2D object-guided episode with the gripper opening and the
arm lifting clear. It is also shown in the [final render](media/v2d_full_release_final.png).
The original full-resolution MOV and licensed MANO models are not committed.
An authorized [short source-video excerpt](media/bowl_to_plate_demonstration_excerpt.mp4) shows
the recorded task.

The [expert replay metrics](expert_baseline_metrics.json) establish a control:
the same five initial states also succeed with the unmodified official actions.
The video-guided transport takes roughly twice as many control steps because
it follows the recorded six-second motion. Both rows use the expert grasp.
These are controlled replays, not trained policy evaluations, and do
not measure generalization to new objects or tasks.

Environment: LIBERO `8f1084e`, robosuite 1.4.0, MuJoCo 2.3.7, NumPy 1.26.4,
Python 3.10, headless EGL on an NVIDIA L40S. The task is
`pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`.

## Release-visible replay

The first V2D-guided rollout was repeated with `--finish-release` on the same
official initial state. The bowl first reached the physics success condition
at step 190. The expert release command began at step 200; 36 further scripted
open-and-retreat steps left the gripper clear of the bowl. The final gripper
joint positions were `[0.0398, -0.0395]`, close to the initial open positions,
and `env.check_success()` was still true at step 239. This is **1/1 extended
replay**, separate from the five first-success results above. See the
[machine-readable metrics](v2d_full_release_metrics.json).

This extended run used LIBERO `8f1084e`, robosuite 1.4.0, MuJoCo 2.3.7,
NumPy 1.26.4, Python 3.11, and macOS CGL rendering. The user-recorded video
still drives only transport; the grasp, release, and retreat are not inferred
from that video.

## Full video-to-hand retargeting

NVIDIA V2D completed reconstruction of the 243-frame, 10 fps sampled clip
with reference frame 70, HaMeR, an approximate 14 cm bowl mesh, and gravity
alignment. Its bundle reports 243 right-hand and object poses; early hand
poses were filled before the hand entered the camera view. The
object moved about 0.35 m in reconstructed x and about 0.07 m in z over the
clip. [The 9–15 second object track](../data/derived/bowl_to_plate_v2d_object_track.csv)
is shared without the private hand trajectory or licensed MANO weights.

The reconstructed bowl orientation is wrong near placement. The saved mesh's
local +Z axis points toward its open rim, but its estimated world-Z component
changes from +0.92 at frame 120 (12.0 s) to -0.76 at frame 130 (13.0 s), and
remains about -0.96 at frame 150 (15.0 s). The physical bowl stays upright.
The saved object-to-camera pose, camera-to-world pose, and object-to-world pose
are mutually consistent, so this is a pose-estimation failure rather than an
overlay coordinate error. The cause cannot be isolated from this run; hand
occlusion and the approximate handle-free mesh are plausible contributors.
The successful V2D object-guided transport exports only the pose translation,
not this incorrect rotation.

For the [upright-constrained visualization](media/v2d_upright_constrained_stills.png),
[`constrain_v2d_bowl_upright.py`](../scripts/constrain_v2d_bowl_upright.py)
replaces the proxy's rotation with a fixed gravity-up orientation in a separate
bundle. It preserves V2D's translations and hand poses. The
[original projection](media/v2d_reconstruction_stills.png) remains available
to show the actual orientation failure. This display correction does not
alter either LIBERO replay or its reported success rate.

The uncalibrated direct MANO-wrist-to-LIBERO retargeting produced **0/1
success** over a 484-step replay. [Six simulator frames](media/v2d_direct_replay_fail_montage.png)
show that the gripper never acquired the bowl. The wrist-to-tool offset and
workspace yaw were left at identity/default values; the real handled bowl
also differs from the simulator model. This is a failed experiment, not a
trained policy evaluation.

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
- V2D's first reference frame had no visible hand; switching to frame 70
  allowed hand segmentation. Licensed MANO assets were placed in WiLoR and
  HaMeR's expected private weights directories on the GPU host.
