#!/usr/bin/env bash
# Run V2D's end-to-end ego pipeline on a video.
# Requires a GPU host: docker with --runtime=nvidia, or a working host venv
# (see V2D reconstruction/scripts/install_ego_e2e_packages.sh).
#
# Usage:
#   V2D_ROOT=~/v2d_pipeline ./scripts/run_v2d.sh \
#       --video_path data/raw/clip01.mp4 --prompt "red mug" \
#       --output_dir data/outputs/clip01 --depth_source vipe
set -euo pipefail

V2D_ROOT="${V2D_ROOT:-$HOME/v2d_pipeline}"
E2E="$V2D_ROOT/reconstruction/modules/v2d_pipelines/run_v2d_ego_e2e.py"

if [[ ! -f "$E2E" ]]; then
  echo "V2D entry point not found: $E2E" >&2
  echo "Clone https://github.com/Swaguto/NVIDIA-V2D-Challenge and set V2D_ROOT." >&2
  exit 1
fi

exec python "$E2E" "$@"
