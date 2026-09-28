"""Durable long-video job records, with exclusive ownership and content checks."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path

from .media import atomic_output
from .models import sha256_file

SCHEMA_VERSION = 1
PROCESSING_REVISION = "bounded-frames-v1"


def work_directory(source, output_dir, work_dir=None):
    return Path(work_dir or Path(output_dir) / f"FlashVSR_{Path(source).stem}_Final.mp4.job").expanduser().resolve()


def file_identity(path):
    path = Path(path)
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError(f"File changed while hashing: {path}")
    return {"bytes": after.st_size, "sha256": digest}


class JobRecord:
    def __init__(self, directory, specification, resume):
        self.directory = Path(directory).expanduser().resolve()
        self.path = self.directory / "job.json"
        self.specification = specification
        self.resume = resume
        self.data = None

    def owned_path(self, relative):
        path = self.directory / relative
        if path.is_symlink() or not path.resolve().is_relative_to(self.directory):
            raise ValueError(f"Job path escapes its directory: {path}")
        return path

    def segment_path(self, index):
        self.owned_path(f"segments/{index:06d}.mkv.json")
        return self.owned_path(f"segments/{index:06d}.mkv")

    @contextmanager
    def open(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        lock_path = self.owned_path(".lock")
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Job is already running: {self.directory}") from error
            try:
                self.owned_path("job.json")
                total = self.specification["input_media"]["frames"]
                limit = self.specification["segment_frames"]
                plan = [{"index": index, "start": start, "frames": min(limit, total-start), "status": "pending"}
                        for index, start in enumerate(range(0, total, limit))]
                if self.resume:
                    if not self.path.is_file():
                        raise FileNotFoundError(f"No job to resume: {self.path}")
                    self.data = json.loads(self.path.read_text(encoding="utf-8"))
                    if self.data.get("schema_version") != SCHEMA_VERSION or self.data.get("specification") != self.specification:
                        raise ValueError("Cannot resume: input, settings, models or runtime changed; use a new --work-dir")
                    actual = self.data.get("segments", [])
                    if [{key: item.get(key) for key in ("index", "start", "frames")} for item in actual] != [
                            {key: item[key] for key in ("index", "start", "frames")} for item in plan]:
                        raise ValueError("Invalid job segment plan")
                else:
                    if any(path.name != ".lock" for path in self.directory.iterdir()):
                        raise FileExistsError(f"Job directory is not empty: {self.directory}; use --resume or a new --work-dir")
                    self.data = {"schema_version": SCHEMA_VERSION, "specification": self.specification,
                                 "created_at": datetime.now(timezone.utc).isoformat(), "status": "pending", "segments": plan}
                    self.save()
                self.owned_path("segments").mkdir(exist_ok=True)
                yield self
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def save(self):
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        with atomic_output(self.path) as temporary:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(self.data, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
