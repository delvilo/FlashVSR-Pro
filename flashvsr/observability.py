"""Leveled logs and versioned per-run reports shared by every entry point."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import sys
import time
import uuid

from . import __version__
from .media import atomic_output

run_id = ContextVar("flashvsr_run_id", default="-")


class RunFilter(logging.Filter):
    def filter(self, record):
        record.run_id = run_id.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = {"time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                "level": record.levelname, "logger": record.name,
                "run_id": getattr(record, "run_id", run_id.get()), "message": record.getMessage()}
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def configure_logging(level="INFO", log_file=None):
    root = logging.getLogger()
    # Replace only handlers owned by this CLI; embedding applications keep theirs.
    for handler in list(root.handlers):
        if getattr(handler, "_flashvsr", False):
            root.removeHandler(handler)
            handler.close()
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(levelname)s [%(run_id)s] %(message)s"))
    handlers = [console]
    if log_file:
        path = Path(log_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(JsonFormatter())
        handlers.append(handler)
    for handler in handlers:
        handler._flashvsr = True
        handler.addFilter(RunFilter())
        root.addHandler(handler)
    root.setLevel(level)
    logging.captureWarnings(True)


def write_report(path, data):
    with atomic_output(path) as temporary:
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def environment():
    versions = {}
    for name in ("torch", "numpy", "pillow", "einops", "safetensors", "imageio", "imageio-ffmpeg",
                 "block-sparse-attn", "flash-attn", "sageattention"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"application": __version__, "python": platform.python_version(), "packages": versions}


class RunReport:
    def __init__(self, config, source, destination, model_identity):
        self.started = time.perf_counter()
        self.data = {"schema_version": 1, "run_id": uuid.uuid4().hex,
                     "started_at": datetime.now(timezone.utc).isoformat(), "status": "running",
                     "mode": config.mode, "parameters": config.as_dict(), "input": str(source),
                     "output": str(destination), "models": model_identity,
                     "environment": environment(), "timings": {}}

    @contextmanager
    def stage(self, name, synchronize=None):
        self.data["stage"] = name
        if synchronize:
            synchronize()
        started = time.perf_counter()
        try:
            yield
            if synchronize:
                synchronize()
        finally:
            self.data["timings"][name] = time.perf_counter() - started

    def finish(self, error=None):
        self.data["total_seconds"] = time.perf_counter() - self.started
        self.data["status"] = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed" if error else "ok"
        if error:
            self.data["error"] = {"type": type(error).__name__, "message": str(error)}
        inference = self.data["timings"].get("inference", 0)
        self.data["inference_seconds"] = inference
        if not error and self.data.get("frames"):
            self.data["inference_fps"] = self.data["frames"] / inference if inference > 0 else None
            self.data["end_to_end_fps"] = self.data["frames"] / self.data["total_seconds"]
        return self.data
