"""Batch and segmented video orchestration over the same InferenceEngine."""

from dataclasses import replace
import logging
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

from .config import VIDEO_SUFFIXES, output_path, validate_input, validate_report_path
from .engine import InferenceEngine
from .media import MediaTools, atomic_output
from .observability import write_report

logger = logging.getLogger(__name__)


def run_batch(config, input_dir, output_dir, metrics_json=None):
    source, destination = Path(input_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Input directory not found: {source}")
    if source == destination or source in destination.parents:
        raise ValueError("Batch output must be outside the input directory")
    inputs = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
    if not inputs:
        raise ValueError(f"No eligible videos in {source}")
    outputs = [output_path(path, destination / path.relative_to(source).parent, config, directory=True) for path in inputs]
    engine = InferenceEngine(config)
    report_path = validate_report_path(metrics_json or destination / "batch.json", *inputs, *outputs,
                                       *(Path(f"{path}.json") for path in outputs),
                                       *(engine.registry.directory / item.name for item in engine.registry.entries))
    report = {"schema_version": 1, "workflow": "batch", "parameters": config.as_dict(), "runs": []}
    started = time.perf_counter()
    try:
        for path, target in zip(inputs, outputs):
            try:
                report["runs"].append(engine.run(path, target))
            except Exception as error:
                logger.error("Batch input failed: %s: %s", path, error)
                report["runs"].append({"input": str(path), "output": str(target), "status": "failed",
                                       "report": f"{target}.json", "error": str(error)})
        failures = sum(item["status"] != "ok" for item in report["runs"])
        report["status"] = "failed" if failures else "ok"
        logger.info("Batch completed: %s succeeded, %s failed", len(inputs)-failures, failures)
        return 1 if failures else 0
    finally:
        report.setdefault("status", "cancelled")
        report["total_seconds"] = time.perf_counter()-started
        write_report(report_path, report)


def run_long(config, source, output_dir, segment_time=60.0, keep_temp=False, metrics_json=None):
    if not math.isfinite(segment_time) or segment_time <= 0:
        raise ValueError("Segment duration must be finite and positive")
    source = validate_input(source)
    if not source.is_file():
        raise ValueError("The long-video worker requires a video file")
    destination = Path(output_dir).expanduser().resolve() / f"FlashVSR_{source.stem}_Final.mp4"
    if destination == source:
        raise ValueError("Input and output must be different files")
    engine = InferenceEngine(replace(config, keep_audio=False))
    report_path = validate_report_path(metrics_json or f"{destination}.json", source, destination,
                                       *(engine.registry.directory / item.name for item in engine.registry.entries))
    media = MediaTools()
    original = media.verify_video(source)
    width, height = round(original["width"] * config.scale), round(original["height"] * config.scale)
    if min(width, height) < 1:
        raise ValueError("Scaled dimensions must be at least one pixel")
    # Validate all weights before splitting. Segment pipelines have independent caches.
    engine.media = media
    engine.registry.check(config.mode)
    work = Path(tempfile.mkdtemp(prefix="flashvsr-segments-"))
    logger.info("Temporary directory: %s", work)
    started = time.perf_counter()
    report = {"schema_version": 1, "workflow": "long", "input": str(source), "output": str(destination),
              "parameters": config.as_dict(), "segment_seconds": segment_time, "runs": [],
              "models": engine.registry.identity(config.mode)}
    try:
        splits, processed = work / "splits", work / "processed"
        splits.mkdir()
        processed.mkdir()
        # Keyframe-aligned copying preserves every decoded input frame.
        media.run(["-i", source, "-map", "0:v:0", "-an", "-c:v", "copy", "-segment_time", str(segment_time),
                   "-f", "segment", "-reset_timestamps", "1", splits / "segment_%06d.mp4"])
        parts = sorted(splits.glob("*.mp4"))
        if not parts:
            raise RuntimeError("FFmpeg produced no segments")
        input_frames, outputs = 0, []
        for index, part in enumerate(parts):
            info = media.verify_video(part)
            input_frames += info["frames"]
            output = processed / part.name
            logger.info("Processing segment %s/%s", index+1, len(parts))
            try:
                report["runs"].append(engine.run(part, output))
            except BaseException:
                # Keep a failed segment's diagnostics when its temporary media is removed.
                child_report = Path(f"{output}.json")
                if child_report.is_file():
                    report["runs"].append(json.loads(child_report.read_text()))
                raise
            media.verify_video(output, width=width, height=height, frames=info["frames"],
                               fps=config.fps or original["fps"], audio_streams=0)
            outputs.append(output)
        if input_frames != original["frames"]:
            raise RuntimeError(f"Splitting changed frame count: {original['frames']} -> {input_frames}")
        # Publish only after the entire concatenated output (and original audio) verifies.
        joined = work / "joined.mp4"
        media.concat_videos(outputs, joined)
        with atomic_output(destination) as temporary:
            if config.keep_audio:
                media.mux_audio(joined, source, temporary)
            else:
                shutil.copyfile(joined, temporary)
            report.update(media.verify_video(temporary, width=width, height=height, frames=original["frames"],
                                              fps=config.fps or original["fps"],
                                              audio_streams=original["audio_streams"] if config.keep_audio else 0))
        report["status"] = "ok"
        logger.info("Verified merged output: %s", destination)
        return 0
    except BaseException as error:
        report.update(status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        report["total_seconds"] = time.perf_counter()-started
        report["peak_allocated_bytes"] = max((item.get("peak_allocated_bytes", 0) for item in report["runs"]), default=0)
        try:
            write_report(report_path, report)
        finally:
            if keep_temp:
                logger.info("Kept temporary files: %s", work)
            else:
                shutil.rmtree(work)
