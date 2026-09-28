"""Bounded frame processing, durable checkpoints, and one final encode."""

from dataclasses import replace
import logging
import math
from pathlib import Path
import time

from .config import validate_input, validate_report_path
from .engine import InferenceEngine
from .jobs import JobRecord, PROCESSING_REVISION, file_identity, work_directory
from .media import MediaError, MediaTools
from .observability import environment, write_report
from .streaming import read_video_frames

logger = logging.getLogger(__name__)


def run_long(config, source, output_dir, segment_time=60.0, keep_temp=False, metrics_json=None,
             *, segment_frames=129, work_dir=None, resume=False, checkpoint=None):
    if not math.isfinite(segment_time) or segment_time <= 0:
        raise ValueError("Segment duration must be finite and positive")
    if type(segment_frames) is not int or segment_frames < 1:
        raise ValueError("Segment frame limit must be a positive integer")
    source = validate_input(source)
    if not source.is_file():
        raise ValueError("The long-video worker requires a video file")
    destination = Path(output_dir).expanduser().resolve() / f"FlashVSR_{source.stem}_Final.mp4"
    if destination == source:
        raise ValueError("Input and output must be different files")
    work = work_directory(source, output_dir, work_dir)
    engine = InferenceEngine(replace(config, keep_audio=False))
    weights = [engine.registry.directory / item.name for item in engine.registry.entries]
    for path in (source, destination, *weights):
        if path == work or work in path.parents:
            raise ValueError("Job directory must not contain the input, final output or model weights")
    report_path = validate_report_path(metrics_json or f"{destination}.json", source, destination, *weights)
    if report_path == work or work in report_path.parents:
        raise ValueError("Metrics/logs must be outside the job directory")
    media = MediaTools()
    original = media.verify_video(source)
    width, height = round(original["width"] * config.scale), round(original["height"] * config.scale)
    if min(width, height) < 1:
        raise ValueError("Scaled dimensions must be at least one pixel")
    limit = min(segment_frames, max(1, math.floor(segment_time * original["fps"])))
    fps = config.fps or original["fps"]
    logger.info("Job directory: %s; at most %s input frames per segment", work, limit)
    logger.info("Hashing input for resume validation: %s", source)
    source_stat = source.stat()
    specification = {"revision": PROCESSING_REVISION, "input": str(source), "input_identity": file_identity(source),
                     "input_media": original, "output": str(destination), "parameters": config.as_dict(),
                     "models": engine.registry.identity(config.mode), "runtime": environment(),
                     "segment_frames": limit, "segment_seconds": segment_time}
    engine.media = media
    expected = dict(width=width, height=height, fps=fps, audio_streams=0, video_codec="ffv1")
    final_expected = dict(width=width, height=height, frames=original["frames"], fps=fps,
                          audio_streams=original["audio_streams"] if config.keep_audio else 0)
    with JobRecord(work, specification, resume).open() as job:
        started = time.perf_counter()
        report = {"schema_version": 1, "workflow": "long", "input": str(source), "output": str(destination),
                  "parameters": config.as_dict(), "segment_seconds": segment_time, "segment_frames": limit,
                  "work_dir": str(work), "resumed": resume, "segments_reused": 0, "segments_processed": 0,
                  "models": specification["models"], "ffmpeg_version": media.ffmpeg_version}
        active = None
        try:
            if resume and job.data.get("status") == "ok" and destination.is_file():
                if file_identity(destination) == job.data.get("output_identity"):
                    report.update(media.verify_video(destination, **final_expected), status="ok")
                    report["segments_reused"] = len(job.data["segments"])
                    logger.info("Completed job and final output verified; nothing to process")
                    return 0
            job.data.update(status="running")
            job.data.pop("error", None)
            for item in job.data["segments"]:
                target = job.segment_path(item["index"])
                valid = False
                if item["status"] == "done" and target.is_file():
                    try:
                        valid = file_identity(target) == item.get("identity")
                        if valid:
                            media.verify_video(target, frames=item["frames"], **expected)
                    except (OSError, MediaError):
                        valid = False
                if valid:
                    report["segments_reused"] += 1
                else:
                    item["status"] = "pending"
                    item.pop("report", None)
            job.save()
            if checkpoint is not None:
                checkpoint(job)
            pending = [item for item in job.data["segments"] if item["status"] != "done"]
            if pending:
                # Verify weights once before decoding, then retain one model until
                # every pending segment has finished. Each call resets video caches.
                with engine.session(), read_video_frames(media, source, original, pending[0]["start"]) as reader:
                    for item in job.data["segments"][pending[0]["index"]:]:
                        if item["status"] == "done":
                            reader.discard(item["frames"])
                            continue
                        active = item
                        item.update(status="running")
                        item.pop("error", None)
                        job.save()
                        logger.info("Processing segment %s/%s (frames %s–%s)", item["index"]+1,
                                    len(job.data["segments"]), item["start"], item["start"]+item["frames"]-1)
                        frames = reader.read(item["frames"])
                        try:
                            target = job.segment_path(item["index"])
                            item["report"] = engine.run(source, target, input_frames=frames,
                                                        media_info={**original, "frames": item["frames"]},
                                                        frame_start=item["start"], lossless=True)
                        finally:
                            del frames
                        media.verify_video(target, frames=item["frames"], **expected)
                        item.update(status="done", identity=file_identity(target))
                        report["segments_processed"] += 1
                        job.save()
                        active = None
                        if checkpoint is not None:
                            checkpoint(job)
            after = source.stat()
            if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (source_stat.st_size, source_stat.st_mtime_ns, source_stat.st_ctime_ns):
                raise RuntimeError("Input changed while processing; start a new job")
            outputs = [job.segment_path(item["index"]) for item in job.data["segments"]]
            job.data["status"] = "encoding"
            job.save()
            device_index = int(config.device.split(":")[1]) if ":" in config.device else 0
            report.update(media.encode_segments(outputs, destination, fps=fps, quality=config.quality,
                                               audio_source=source if config.keep_audio else None,
                                               device_index=device_index))
            media.verify_video(destination, **final_expected)
            job.data.update(status="ok", output_identity=file_identity(destination))
            job.save()
            report["status"] = "ok"
            if not keep_temp:
                for path in outputs:
                    try:
                        path.unlink(missing_ok=True)
                        Path(f"{path}.json").unlink(missing_ok=True)
                    except OSError:
                        logger.warning("Could not remove completed cache: %s", path, exc_info=True)
            logger.info("Verified merged output: %s", destination)
            return 0
        except BaseException as error:
            status = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
            if active is not None:
                active.update(status=status, error=str(error))
            job.data.update(status=status, error=str(error))
            job.save()
            report.update(status=status, error=str(error))
            logger.info("Completed segments retained. Resume with the same options and --resume: %s", work)
            raise
        finally:
            report["runs"] = [item["report"] for item in job.data["segments"] if "report" in item]
            report["total_seconds"] = time.perf_counter()-started
            report["model_loads"] = engine.model_loads
            report["peak_allocated_bytes"] = max((item.get("peak_allocated_bytes", 0) for item in report["runs"]), default=0)
            try:
                write_report(report_path, report)
            except OSError:
                if report.get("status") == "ok":
                    raise
                logger.exception("Could not write long-job failure report: %s", report_path)
