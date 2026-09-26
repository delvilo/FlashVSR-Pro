"""Command-line validation that does not import the inference stack."""

import argparse
import math
from pathlib import Path
import re
import sys

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
OUTPUT_SUFFIXES = VIDEO_SUFFIXES | {".gif"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MODES = ("full", "tiny", "tiny-long")


def run_cli(operation, argv=None):
    """Shared exit codes: 0 success, 1 failure, 2 usage, 130 interruption."""
    try:
        return operation(argv) or 0
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def cuda_device(value):
    if not re.fullmatch(r"cuda(?::\d+)?", value):
        raise argparse.ArgumentTypeError("use cuda or cuda:N; the sparse backend requires CUDA")
    return value


def segment_seconds(value):
    try:
        parts = value.split(":")
        if len(parts) == 1:
            return positive_float(value)
        if len(parts) != 3:
            raise ValueError
        hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
        if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
            raise ValueError
        return positive_float(str(hours * 3600 + minutes * 60 + seconds))
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError("use positive seconds or HH:MM:SS") from error


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="FlashVSR-Pro inference on Linux / Google Colab")
    parser.add_argument("-i", "--input", required=True, help="Video file or PNG/JPEG directory")
    parser.add_argument("-o", "--output", default="./results", help="Output directory or video/GIF path")
    parser.add_argument("--mode", choices=MODES, default="tiny")
    parser.add_argument("--tile-dit", action="store_true")
    parser.add_argument("--tile-vae", action="store_true", help="Tiled VAE decoding (full mode only)")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=24)
    parser.add_argument("--keep-audio", action="store_true", help="Preserve all input audio tracks")
    parser.add_argument("--scale", type=positive_float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparse-ratio", type=positive_float, default=2.0)
    parser.add_argument("--kv-ratio", type=positive_float, default=3.0)
    parser.add_argument("--local-range", type=int, default=11)
    parser.add_argument("--color-fix", action="store_true")
    parser.add_argument("--fps", type=positive_float, help="Output FPS; defaults to the input rate")
    parser.add_argument("--quality", type=int, choices=range(11), default=10, metavar="0..10")
    parser.add_argument("--device", type=cuda_device, default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--metrics-json", type=Path, help="Write GPU timing, memory and output metadata")
    args = parser.parse_args(argv)
    if args.local_range <= 0:
        parser.error("--local-range must be greater than zero")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    if args.tile_size < 128 or args.tile_size % 32:
        parser.error("--tile-size must be a multiple of 32 and at least 128")
    if not 0 <= args.overlap < args.tile_size // 2:
        parser.error("--overlap must be nonnegative and less than half --tile-size")
    if args.tile_vae and args.mode != "full":
        parser.error("--tile-vae is supported only with --mode full")
    return args


def validate_input(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        if not any(p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in path.iterdir()):
            raise ValueError(f"No PNG/JPEG frames found in {path}")
    elif not path.is_file():
        raise FileNotFoundError(f"Input not found: {path}")
    elif path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported input format: {path.suffix}")
    return path


def validate_output(path, input_path=None, keep_audio=False):
    path = Path(path).expanduser().resolve()
    if path.is_dir() or not path.suffix:
        return path
    if path.suffix.lower() not in OUTPUT_SUFFIXES:
        raise ValueError(f"Unsupported output format: {path.suffix}")
    if input_path is not None and path == Path(input_path).resolve():
        raise ValueError("Input and output must be different files")
    if keep_audio and path.suffix.lower() == ".gif":
        raise ValueError("GIF cannot preserve audio; select a video output")
    return path
