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

A task-matched official demonstration provides the grasp and release. This
solves the mismatch between the real handled bowl and LIBERO's handle-free bowl,
but it limits the claim: the real video controls transport, not the entire
manipulation. The sparse waypoints were inspected manually because the hand
occludes the bowl in several frames. The result is reproducible with the
committed CSV and script; see [results](results.md).

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
- Hand occlusion can shorten the valid trajectory. Interpolating across a long
  gap could invent an unsafe motion, so the extractor uses the longest valid
  run.

The live V2D run uses an approximate metric bowl mesh. Object-pose and replay
quality must be inspected before making any claim about this route. A failed
replay is a calibration result, not a training demonstration.

## What would establish a policy benefit

Train the same policy architecture with official LIBERO data alone and with
additional successful video-derived episodes. Evaluate both on the same held
out initial states and report task success with uncertainty. This comparison
has not been run; replay success alone does not imply a policy improvement.
