# Design choices

Mapped to the rubric's questions: *creativity*, *does your data play a real
role*, *performance of the policy / quality of world-model predictions*.

## 1. The central bet: motion in, pixels stay synthetic

Two naive ways to use a phone video for Libero:

- **Feed real frames to the policy** — huge visual domain gap (phone camera vs
  libero renderer); a fine-tuned VLA will latch onto backgrounds, lighting,
  camera intrinsics. Fails in sim eval.
- **Use only the task language** — your data plays no real role; the rubric
  explicitly checks this.

We take the third path: **the video contributes the motion distribution, the
simulator contributes the pixels.** V2D recovers *what the hand did* (wrist SE3
trajectory, aperture timing); we replay it through Libero's physics to obtain
observations that are rendered in-domain. The result: demos that carry the
idiosyncrasies of a real human grasp (approach angle, hesitation, closure
timing) without a single real pixel entering the training set.

## 2. Why V2D (vs. off-the-shelf hand tracking)

- **MANO + world frame in one pipeline**: ViPE camera pose + Dyn-HaMR hand
  reconstruction gives wrist trajectories in a gravity-aligned world (GeoCalib),
  which is exactly the input a robot retargeter needs.
- **Object tracking too**: FoundationPose gives the object pose for the same
  frames — used later for grasp-point verification and for placing the object
  in the Libero scene consistently with where the hand reached.
- **Reproducible contract**: `result.npz` is a documented, versioned bundle —
  the adapter (`extract.py`) is a pure consumer; nothing about our code depends
  on V2D internals.

## 3. The scale problem, and why `grasp_aperture` is the right anchor

Monocular depth (ViPE/MoGe) is not guaranteed metric. Options we rejected:

| anchor | problem |
|---|---|
| known object width via FoundationPose mesh | SAM3D meshes are not metric; scale entangled with pose scale |
| hand size from MANO betas | betas fitted in the same non-metric world — circular |
| depth reading at object | monocular depth scale drifts frame to frame |

**Chosen:** during the recording the hand *holds the measured object* — the
aperture while holding ≈ object width in whatever units the pipeline produced.
`scale = width_measured / median(aperture_in_hold_window)`. One measurement,
robust to depth drift, and the hold window is easy to find (aperture minimum
region). The sim replay is the backstop: a wrong scale shows up immediately as
a missed grasp and the episode is filtered out.

## 4. Gripper signal from fingertip distance

No grasp labels, no force data: thumb–index vertex distance (MANO tip verts
from V2D's own track3 code) normalized to [0,1]. Continuous during the
trajectory, thresholded implicitly by the sim (physics decides). Invalid frames
are forward-filled; only the longest valid run is replayed (interpolating
through occlusion would invent motion).

## 5. Replay as physics-based filtering

The retargeted trajectory is executed through Libero's OSC controller. Only
rollouts where `env.check_success()` passes become training demos. This makes
the pipeline self-correcting: bad scale, bad frame alignment, or
non-physical hand motion simply fail to produce a demo instead of poisoning the
dataset. Failed-rollout rate is reported (see `results.md`) — it's a direct
measure of retarget quality.

## 6. Policy choice

π0/π0.5 via **openpi** — LIBERO is their canonical fine-tune example
(`pi05_libero` checkpoint, 96.85% on the suite), open weights, real
infrastructure for mixing datasets. **OpenVLA-OFT** as the fast alternative
(97.1%, ~26× inference speedup — also speaks to the rubric's
inference-optimisation hint). Octo deliberately not used: 75.1% leaves no head
room to measure our data's contribution.

Baselines to report:

1. official Libero demos only (baseline),
2. official + our video-derived demos,
3. (ablation) our demos without the success filter.

## 7. What we expect to struggle with (honest list, to be filled with numbers)

- **Workspace alignment**: guessing the V2D→Libero origin/yaw by hand; a wrong
  yaw puts the grasp centimetres off the object. Mitigation: print wrist range
  vs scene object pose, tune `workspace.*`, and iterate with `--save-rejects`.
- **Occlusion gaps**: hand disappears behind the object → longest-run
  truncation shortens demos.
- **Aperture proxy**: thumb–index distance ≠ contact; pinch grasps on thin
  objects may close "too early" in the recorded motion — physics filter
  catches it, but yield drops.
- **Video deadline**: submission window closed before compute was available —
  artifacts will be attached post-hoc if the evaluation allows it.
