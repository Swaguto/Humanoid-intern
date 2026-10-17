# Design choices

The [Humanoid internship challenge](https://jobs.ashbyhq.com/humanoid/e1a2a9de-ad23-4632-9d93-ee50fd41a221)
asks for personally collected data to drive a robotic manipulator in simulation.
The applicant's video shows a black bowl lifted and placed on a plate, matching
a LIBERO spatial task. We tested a narrow video-guided replay and built a more
ambitious full-hand retargeter.

## Tested route: video-guided transport

The personally recorded clip supplies measured bowl centers and timing. During
the transport phase, the controller converts the bowl's horizontal image motion
into progress from the simulator bowl toward the plate. Upward image motion
raises the end-effector slightly, and the time spacing sets the transport
speed. LIBERO's OSC controller executes these targets, and physics determines
whether the bowl stays grasped and reaches the plate.

A task-matched official demonstration provides the grasp and other actions
outside video-guided transport. This addresses the mismatch between the real
handled bowl and LIBERO's handle-free bowl,
but it limits the claim: the real video controls transport, not the entire
manipulation. The sparse waypoints were inspected manually because the hand
occludes the bowl in several frames. The result is reproducible with the
committed CSV and script; see [results](results.md).

The second tested variant uses V2D's reconstructed 3D bowl x-coordinate for
transport progress and its height above a straight start-to-end trend for lift.
The lift returns to zero at handoff to the expert post-transport actions. It
also succeeded on the same five initial states, but still depends on the
expert grasp. The rollout ends at physics success before gripper release.
Its published CSV contains only object positions from the private
V2D bundle.

The output HDF5 contains simulator camera images and actions from the executed
rollout. It is suitable as a candidate demonstration for later training, but
we have not trained or evaluated a learned policy with it. The five successful
replays use official initial states for the same task and therefore do not test
out-of-distribution generalization.

## Experimental route: full-hand retargeting

The `v2d_to_libero` package consumes NVIDIA V2D's `result.npz`: MANO wrist pose,
thumb-index aperture, and object pose. It aligns the reconstructed object to
the simulator object, converts wrist pose to end-effector targets, resamples to
LIBERO's control frequency, and translates target poses into OSC delta actions.
Only a rollout passing `env.check_success()` is saved as a training episode.

This route has several unresolved calibration risks:

- Monocular reconstruction may not have reliable metric scale. The default
  scale is explicit and must be checked against the recorded object.
- A single object point aligns translation but does not determine world yaw or
  wrist-to-tool offset.
- Real hand aperture is only a proxy for a two-finger robot gripper command.
- The video bowl's handle has no simulator counterpart; a faithful wrist path
  may produce the wrong contact geometry.
- V2D filled early frames before the hand entered the view and marked them
  valid in its result bundle. The direct replay trusted those filled poses;
  a successful full-hand system needs visibility checks and explicit trimming
  of such intervals.

The completed V2D run used an approximate metric bowl mesh. The first direct
hand retargeting replay failed because the gripper missed the bowl. The
default identity wrist-to-tool transform and workspace yaw were not calibrated;
the failed replay is a calibration result, not a training demonstration.

## What would establish a policy benefit

Train the same policy architecture with official LIBERO data alone and with
additional successful video-derived episodes. Evaluate both on the same held
out initial states and report task success with uncertainty. This comparison
has not been run; replay success alone does not imply a policy improvement.
