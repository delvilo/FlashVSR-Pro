"""Split, process and verify a long video before publishing the merged output."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from utils.cli import MODES, positive_float, run_cli, segment_seconds, validate_input
from utils.media import MediaTools

PROJECT_DIR = Path(__file__).resolve().parent


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output-dir", "--output_dir", dest="output_dir", required=True, type=Path)
    parser.add_argument("--segment-time", "--segment_time", dest="segment_time", type=segment_seconds, default=60.0)
    parser.add_argument("--mode", choices=MODES, default="tiny")
    parser.add_argument("--scale", type=positive_float, default=2.0)
    parser.add_argument("--keep-temp", action="store_true", help="Keep this run's temporary segments for diagnosis")
    args = parser.parse_args(argv)
    source = validate_input(args.input)
    if not source.is_file():
        raise ValueError("The long-video worker requires a video file")
    destination = args.output_dir.expanduser().resolve() / f"FlashVSR_{source.stem}_Final.mp4"
    if source == destination:
        raise ValueError("Input and output must be different files")
    media = MediaTools()
    original = media.verify_video(source)
    width, height = round(original['width'] * args.scale), round(original['height'] * args.scale)
    if min(width, height) < 1:
        raise ValueError("Scaled dimensions must be at least one pixel")
    work = Path(tempfile.mkdtemp(prefix="flashvsr-segments-"))
    print(f"Temporary directory: {work}", flush=True)
    try:
        splits, processed = work / "splits", work / "processed"
        splits.mkdir()
        processed.mkdir()
        media.run(["-i", source, "-map", "0:v:0", "-map", "0:a?", "-c", "copy",
                   "-segment_time", str(args.segment_time), "-f", "segment",
                   "-reset_timestamps", "1", splits / "segment_%06d.mp4"])
        parts = sorted(splits.glob("*.mp4"))
        if not parts:
            raise RuntimeError("FFmpeg produced no segments")
        input_frames = 0
        outputs = []
        for index, part in enumerate(parts):
            info = media.verify_video(part)
            input_frames += info['frames']
            output = processed / part.name
            print(f"Processing segment {index + 1}/{len(parts)}", flush=True)
            result = subprocess.run([
                sys.executable, str(PROJECT_DIR / "infer.py"), "-i", str(part), "-o", str(output),
                "--mode", args.mode, "--scale", str(args.scale), "--keep-audio",
            ])
            if result.returncode in (130, -2):
                raise KeyboardInterrupt
            if result.returncode:
                raise RuntimeError(f"Segment {index + 1} inference exited {result.returncode}")
            media.verify_video(output, width=width, height=height, frames=info['frames'],
                               fps=info['fps'], audio_streams=info['audio_streams'])
            outputs.append(output)
        if input_frames != original['frames']:
            raise RuntimeError(f"Splitting changed frame count: {original['frames']} -> {input_frames}")
        media.concat_videos(outputs, destination)
        print(f"Done! Output: {destination}")
        return 0
    finally:
        if args.keep_temp:
            print(f"Kept temporary files: {work}")
        else:
            shutil.rmtree(work)


def main(argv=None):
    return run_cli(_main, argv)


if __name__ == "__main__":
    raise SystemExit(main())
