"""One CLI for inference, batch processing, long videos and model management."""

import argparse
import json
import logging
import math
from pathlib import Path
import sys

from .config import InferenceConfig, MODES, VIDEO_SUFFIXES, output_path, validate_report_path
from .observability import configure_logging

logger = logging.getLogger(__name__)


def run_cli(operation, argv=None):
    """Exit codes: 0 success, 1 operational failure, 2 usage, 130 interruption."""
    try:
        return operation(argv) or 0
    except KeyboardInterrupt:
        logger.warning("Cancelled")
        return 130
    except Exception as error:
        logger.error("%s", error, exc_info=logger.isEnabledFor(logging.DEBUG))
        return 1


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


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
        return positive_float(str(hours*3600 + minutes*60 + seconds))
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError("use positive seconds or HH:MM:SS") from error


def model_options(parser):
    parser.add_argument("--model-version", choices=("v1.1",), default="v1.1")
    parser.add_argument("--model-dir", type=Path, help="Explicit model directory; otherwise use the shared versioned cache")


def log_options(parser):
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    parser.add_argument("--log-file", type=Path, help="Append structured JSONL logs")


def inference_options(parser):
    parser.add_argument("--mode", choices=MODES, default="tiny")
    parser.add_argument("--scale", type=positive_float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparse-ratio", type=positive_float, default=2.0)
    parser.add_argument("--kv-ratio", type=positive_float, default=3.0)
    parser.add_argument("--local-range", type=int, default=11)
    parser.add_argument("--color-fix", action="store_true")
    parser.add_argument("--fps", type=positive_float)
    parser.add_argument("--quality", type=int, choices=range(11), default=10, metavar="0..10")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--tile-dit", action="store_true")
    parser.add_argument("--tile-vae", action="store_true", help="Tiled VAE decoding (full mode only)")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=24)
    parser.add_argument("--keep-audio", action="store_true", help="Preserve every original audio track")
    parser.add_argument("--metrics-json", type=Path, help="Override report path; defaults to OUTPUT.json or batch.json")
    model_options(parser)
    log_options(parser)


def parser():
    root = argparse.ArgumentParser(prog="flashvsr", description="FlashVSR-Pro on native Linux / Google Colab")
    commands = root.add_subparsers(dest="command", required=True)
    single = commands.add_parser("infer", help="Upscale one video or image directory")
    single.add_argument("-i", "--input", required=True)
    single.add_argument("-o", "--output", default="./results")
    batch = commands.add_parser("batch", help="Recursively process a video directory")
    batch.add_argument("--input-dir", type=Path, default=Path("inputs"))
    batch.add_argument("--output-dir", type=Path, default=Path("results"))
    long = commands.add_parser("long", help="Split, process and merge a long video")
    long.add_argument("-i", "--input", required=True)
    long.add_argument("-o", "--output-dir", "--output_dir", required=True, type=Path)
    long.add_argument("--segment-time", "--segment_time", type=segment_seconds, default=60.0)
    long.add_argument("--keep-temp", action="store_true")
    for command in (single, batch, long):
        inference_options(command)
    models = commands.add_parser("models", help="List, verify or download versioned model weights")
    actions = models.add_subparsers(dest="action", required=True)
    for name in ("list", "check", "download"):
        action = actions.add_parser(name)
        action.add_argument("--mode", choices=("all", *MODES), default="all")
        model_options(action)
        log_options(action)
    return root


def parse_args(argv=None):
    root = parser()
    args = root.parse_args(argv)
    if args.command != "models":
        try:
            InferenceConfig.from_namespace(args)
        except ValueError as error:
            root.error(str(error))
    return args


def _main(argv=None):
    args = parse_args(argv)
    if args.command == "models":
        # Model CLI needs only the Python standard library.
        from .models import ModelRegistry
        registry = ModelRegistry(args.model_version, args.model_dir)
        if args.log_file:
            validate_report_path(args.log_file, *(registry.directory / item.name for item in registry.entries))
        configure_logging(args.log_level, args.log_file)
        if args.action == "list":
            print(json.dumps({**registry.identity(args.mode), "files": [
                {"name": item.name, "bytes": item.bytes, "sha256": item.sha256, "modes": item.modes}
                for item in registry.selected(args.mode)]}, indent=2))
        elif args.action == "download":
            registry.download(args.mode)
        else:
            results = registry.report(args.mode)
            print(json.dumps(results, indent=2))
            return int(any(item["status"] != "ok" for item in results))
        return 0
    config = InferenceConfig.from_namespace(args)
    # Protect media and reports from an accidentally overlapping log filename.
    if args.log_file:
        from .models import ModelRegistry
        registry = ModelRegistry(args.model_version, args.model_dir)
        protected = [args.metrics_json, *(registry.directory / item.name for item in registry.entries)]
        if args.command == "infer":
            target = output_path(args.input, args.output, config)
            protected += [args.input, target, f"{target}.json"]
        elif args.command == "long":
            target = args.output_dir / f"FlashVSR_{Path(args.input).stem}_Final.mp4"
            protected += [args.input, target, f"{target}.json"]
        else:
            inputs = [path for path in args.input_dir.rglob("*") if path.is_file()]
            targets = [output_path(path, args.output_dir / path.relative_to(args.input_dir).parent,
                                   config, directory=True) for path in inputs if path.suffix.lower() in VIDEO_SUFFIXES]
            protected += inputs + targets + [Path(f"{target}.json") for target in targets]
            protected += [args.output_dir / "batch.json"]
        validate_report_path(args.log_file, *protected)
    configure_logging(args.log_level, args.log_file)
    if args.command == "infer":
        from .engine import InferenceEngine
        InferenceEngine(config).run(args.input, args.output, args.metrics_json)
        return 0
    from .workflows import run_batch, run_long
    if args.command == "batch":
        return run_batch(config, args.input_dir, args.output_dir, args.metrics_json)
    return run_long(config, args.input, args.output_dir, args.segment_time, args.keep_temp, args.metrics_json)


def main(argv=None):
    return run_cli(_main, argv)


def legacy_main(command, argv=None):
    return main([command, *(sys.argv[1:] if argv is None else argv)])
