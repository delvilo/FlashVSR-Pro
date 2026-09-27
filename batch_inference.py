"""Process a directory and return failure if any child inference fails."""

import argparse
from pathlib import Path
import subprocess
import sys

from utils.cli import MODES, VIDEO_SUFFIXES, positive_float, run_cli

PROJECT_DIR = Path(__file__).resolve().parent
EXCLUDE_FILES = {"MyOwnSwordsman_S01E01.mp4", "MyOwnSwordsman_S01E01_1080p.mp4"}
SPECIAL_CONFIGS = {"example_audio.mp4": ["--keep-audio"]}


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_DIR / "inputs")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--mode", choices=MODES, default="tiny")
    parser.add_argument("--scale", type=positive_float, default=2.0)
    args = parser.parse_args(argv)
    source = args.input_dir.expanduser().resolve()
    destination = args.output_dir.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Input directory not found: {source}")
    if source == destination or source in destination.parents:
        raise ValueError("Batch output must be outside the input directory")
    files = sorted(path for path in source.rglob("*")
                   if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
                   and path.name not in EXCLUDE_FILES)
    if not files:
        raise ValueError(f"No eligible videos in {source}; supply videos or download samples first")
    failures = []
    for path in files:
        target = destination / path.relative_to(source).parent
        target.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(PROJECT_DIR / "infer.py"), "-i", str(path),
                   "-o", str(target), "--mode", args.mode, "--scale", str(args.scale)]
        command.extend(SPECIAL_CONFIGS.get(path.name, []))
        print(f"Processing: {path.relative_to(source)}", flush=True)
        result = subprocess.run(command)
        if result.returncode in (130, -2):
            raise KeyboardInterrupt
        if result.returncode:
            failures.append(path)
            print(f"Failed (exit {result.returncode}): {path}", file=sys.stderr)
    print(f"Batch completed: {len(files) - len(failures)} succeeded, {len(failures)} failed.")
    return 1 if failures else 0


def main(argv=None):
    return run_cli(_main, argv)


if __name__ == "__main__":
    raise SystemExit(main())
