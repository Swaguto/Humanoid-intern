# Reproduction guide

The [README](../README.md) shows the visual result. This page has the complete commands and the limits of each route.

## Tested video-guided and V2D object-guided transport

Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) and its requirements, then download the `libero_spatial` demonstrations. The replay reads `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5`. The tested setup was LIBERO `8f1084e`, robosuite 1.4.0, MuJoCo 2.3.7, NumPy 1.26.4, Python 3.10, EGL on an NVIDIA L40S.

```bash
export PYTHONPATH=/absolute/path/to/LIBERO:$PWD
export MUJOCO_GL=egl
python scripts/video_guided_replay.py \
  --config configs/target.yaml \
  --expert-hdf5 /absolute/path/to/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 \
  --mode video --episodes 5 --out-dir data/video_guided_replay
```

Run the other variants with `--mode v2d` or `--mode expert`, choosing a separate `--out-dir` for each. The default 2D waypoints are in [`data/derived/bowl_to_plate_waypoints.csv`](../data/derived/bowl_to_plate_waypoints.csv); the V2D 3D positions are in [`data/derived/bowl_to_plate_v2d_object_track.csv`](../data/derived/bowl_to_plate_v2d_object_track.csv). Both replace only expert transport actions. The script writes `metrics.json`, a final render for every attempt, and a project-specific HDF5 episode for each success.

To annotate another video, resize frames to 512×288, record bowl-center pixels and timestamps in the waypoint CSV format, and set `--waypoints`, `--start-s`, and `--end-s`. This requires a visible bowl trajectory and an expert demonstration of the same task. The 3D variant uses V2D object x displacement and measured lift, normalized over the 9–15 second transport segment. Regenerate its CSV from a local V2D bundle with:

```bash
python3 scripts/export_v2d_object_track.py \
  --result-dir /absolute/path/outputs/clip01/result_gravity_aligned \
  --fps 10 --out data/derived/bowl_to_plate_v2d_object_track.csv
```

The five video-guided replay lengths were 190, 206, 201, 191, and 199 steps; the V2D object-guided lengths were 190, 206, 202, 190, and 201. The successful HDF5 files stay local because they contain full simulator camera streams. See [results and machine-readable metrics](results.md).

To render a complete two-camera MP4 from a successful rollout, install
`ffmpeg` and run:

```bash
python3 scripts/render_replay_video.py \
  --input data/v2d_object_replay/v2d_demo_0_ok.hdf5 \
  --output data/v2d_object_replay/full_replay.mp4
```

The example input path assumes the V2D replay used
`--out-dir data/v2d_object_replay`. The [published MP4](media/full_libero_bowl_placement.mp4)
uses all 190 frames of that successful episode at the recorded 20 Hz.

## Experimental full-hand route

The broader route uses **NVIDIA V2D** to estimate a MANO hand trajectory and object pose, maps wrist pose and aperture into LIBERO OSC commands, and filters replays by physics success. V2D reconstructed 243 reported object and hand poses for this 24.3-second, 10 fps clip. Early hand poses were filled before the hand entered the frame. The first direct hand-to-arm replay failed 0/1 because the gripper missed the bowl. Policy fine-tuning has not been run.

### 1. Record

Record the complete task, including placement and release. For the default configuration, move a black bowl from table center onto a plate. Keep the object and hand visible from before grasp through release, and set the actual `video_fps` in `configs/target.yaml`. A pickup-only clip cannot pass this task's success check.

### 2. Reconstruct with V2D

Requires a GPU host, [NVIDIA V2D setup](https://github.com/nvidia-isaac/video_to_data/blob/main/reconstruction/docs/ego_e2e_setup.md), its separate model weights, and licensed MANO assets. Use absolute paths because the wrapper runs from V2D's `reconstruction/` directory:

```bash
V2D_ROOT=/path/to/video_to_data ./scripts/run_v2d.sh \
  --video /absolute/path/clip01_10fps.mp4 --object_prompt "black bowl" \
  --object_mesh "$PWD/data/meshes/approx_black_bowl_14cm.obj" \
  --skip_object_scale_estimation \
  --output_dir /absolute/path/outputs/clip01 --reference_frame 70 \
  --hand_tracking hamer --run_gravity_alignment
```

Choose a reference frame with a visible hand. Frame 0 of this clip had no hand; frame 70 did. The approximate 14 cm mesh omits the real bowl's handle, so pose quality has a known limit. Set `video_fps: 10` in the experiment config for this input.

If the reconstructed bowl flips upside down, create a separate upright proxy
bundle for visualization:

```bash
python3 scripts/constrain_v2d_bowl_upright.py \
  --result-dir /absolute/path/outputs/clip01/result_gravity_aligned \
  --out-dir /absolute/path/outputs/clip01/result_gravity_aligned_upright
```

This assumes a gravity-aligned +Z-up world and a bowl mesh that opens along
local +Z. It replaces the proxy's rotation with a fixed world-upright rotation
while preserving its measured translation and hand poses. Keep the original
bundle for evaluating V2D's raw pose quality; this is not recovered rotation.

### 3. Retarget on CPU

```bash
python3 -m v2d_to_libero.extract \
  --result-dir /absolute/path/outputs/clip01/result_gravity_aligned \
  --mano /path/to/mano/models --fps 10 --summary
```

Edit `configs/target.yaml` for the result directory, FPS, object name, scale, workspace yaw, and wrist-to-tool offset. Aligning one reconstructed object point with one simulator point determines translation but not yaw, metric scale, or gripper offset. Inspect rejected rollouts and calibrate these before data collection. MANO hand aperture is only a proxy for a two-finger gripper.

### 4. Replay and inspect

```bash
python3 scripts/check_conventions.py configs/target.yaml
python3 scripts/replay_libero.py --config configs/target.yaml --save-rejects
# successful episodes: data/demos/*_ok.hdf5
```

Only physics successes should become training examples. The uncalibrated replay missed the bowl and produced no successful episode. [Failure frames](media/v2d_direct_replay_fail_montage.png) and [analysis](results.md) are included.

### 5. Convert for a future policy experiment

The project's HDF5 schema is not OpenPI's RLDS format. In a LeRobot environment:

```bash
python3 scripts/convert_to_lerobot.py --input-dir data/demos \
  --repo-id local/video_libero --root /path/to/lerobot/local/video_libero
```

The converter downsamples 20 Hz replay to 10 Hz and exports both camera views, the 8-D state, 7-D actions, and task language. For a baseline with an identical feature schema, use `python3 scripts/merge_lerobot.py --help`, then follow the [OpenPI LIBERO guide](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md). The converter needs real-data verification, and no fine-tuning or policy evaluation has been run. A policy benefit would require baseline and augmented policies evaluated on the same held-out task states.

## Layout and private assets

- `v2d_to_libero/`: MANO forward kinematics, V2D extraction, calibration, pose retargeting, OSC actions.
- `scripts/`: V2D wrapper, measured-motion replay, direct wrist replay, convention checks, data conversion.
- `configs/target.yaml`: task and calibration values.
- `data/derived/`: public waypoints and 3D object positions.
- `data/meshes/`: approximate tracking mesh.
- `docs/`: [results](results.md), [design choices](design.md), and public media.

The original MOV, full V2D bundle, licensed MANO models, and successful HDF5 camera streams are not committed. The [authorized source excerpt](media/bowl_to_plate_demonstration_excerpt.mp4), derived positions, metrics, and simulator renders are public.
