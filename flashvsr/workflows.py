"""Batch orchestration and the shared long-video entry point."""

import logging
from pathlib import Path
import time

from .config import VIDEO_SUFFIXES, output_path, validate_report_path
from .engine import InferenceEngine
from .long_video import run_long
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
