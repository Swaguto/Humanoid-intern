#!/usr/bin/env python3
"""Render a saved LIBERO rollout as a two-camera MP4 with ffmpeg."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path) as episode:
        agent = episode["observations/images/agentview"]
        wrist = episode["observations/images/wrist"]
        if agent.shape != wrist.shape or agent.ndim != 4 or agent.shape[-1] != 3:
            raise ValueError("expected equal-length RGB agent and wrist camera streams")
        fps = float(episode.attrs["fps"])
        if fps <= 0:
            raise ValueError("episode FPS must be positive")

        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", "1024x576",
            "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output_path),
        ]
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 25)
        except OSError:
            font = ImageFont.load_default()

        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            assert encoder.stdin is not None
            for index in range(len(agent)):
                frame = Image.new("RGB", (1024, 576), "#111820")
                for left, dataset in ((0, agent), (512, wrist)):
                    # LIBERO stores camera pixels upside down relative to display.
                    view = Image.fromarray(np.flipud(dataset[index])).resize(
                        (512, 512), Image.Resampling.LANCZOS
                    )
                    frame.paste(view, (left, 64))
                draw = ImageDraw.Draw(frame)
                draw.text((20, 16), "EXTERNAL CAMERA", font=font, fill="white")
                draw.text((532, 16), "WRIST CAMERA", font=font, fill="white")
                encoder.stdin.write(frame.tobytes())
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("ffmpeg failed to encode the replay")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="successful replay HDF5")
    parser.add_argument("--output", required=True, type=Path, help="MP4 output path")
    args = parser.parse_args()
    render(args.input, args.output)
