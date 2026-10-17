#!/usr/bin/env bash
# Run NVIDIA V2D's current ego reconstruction entry point.
#
# Usage:
#   V2D_ROOT=/path/to/video_to_data ./scripts/run_v2d.sh \
#       --video /abs/path/clip01_10fps.mp4 --object_prompt "black bowl" \
#       --object_mesh /abs/path/approx_black_bowl_14cm.obj \
#       --skip_object_scale_estimation \
#       --output_dir /abs/path/outputs/clip01 --reference_frame 70 \
#       --hand_tracking hamer --run_gravity_alignment
set -euo pipefail

V2D_ROOT="${V2D_ROOT:-$HOME/video_to_data}"
RECON="$V2D_ROOT/reconstruction"
E2E="$RECON/modules/v2d_pipelines/run_ego_reconstruction.py"

if [[ ! -f "$E2E" ]]; then
  echo "V2D entry point not found: $E2E" >&2
  echo "Clone https://github.com/nvidia-isaac/video_to_data and set V2D_ROOT." >&2
  exit 1
fi

cd "$RECON"
exec python3 "$E2E" "$@"
