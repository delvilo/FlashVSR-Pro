"""Versioned, verified model storage, usable without PyTorch or a GPU."""

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
from importlib.resources import files
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

logger = logging.getLogger(__name__)


def model_directory(version="v1.1", directory=None):
    explicit = directory or os.getenv("FLASHVSR_MODEL_PATH") or os.getenv("FLASHVSR-Pro_MODEL_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve()
    root = os.getenv("FLASHVSR_CACHE_DIR")
    if not root:
        root = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "flashvsr"
    return (Path(root).expanduser() / version).resolve()


def sha256_file(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@dataclass(frozen=True)
class ModelFile:
    name: str
    bytes: int
    sha256: str
    modes: tuple[str, ...]
    bundled: bool = False


class ModelRegistry:
    def __init__(self, version="v1.1", directory=None):
        manifest = json.loads(files("flashvsr").joinpath("assets/models.json").read_text())
        if version not in manifest["versions"]:
            raise ValueError(f"Unknown model version: {version}")
        self.version = version
        self.release = manifest["versions"][version]
        self.directory = model_directory(version, directory)
        self.entries = tuple(ModelFile(**item) for item in self.release["files"])
        self._verified = {}

    def selected(self, mode="all"):
        if mode not in ("all", "full", "tiny", "tiny-long"):
            raise ValueError(f"Unsupported inference mode: {mode}")
        return tuple(item for item in self.entries if mode == "all" or mode in item.modes)

    def verify(self, item, path=None, *, force=False):
        path = Path(path) if path is not None else self.directory / item.name
        if not path.is_file():
            raise FileNotFoundError(f"Model weight missing: {path}. Run: flashvsr models download --mode all")
        stat = path.stat()
        signature = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, item.sha256)
        if not force and self._verified.get(item.name) == signature:
            return path
        if stat.st_size != item.bytes:
            raise ValueError(f"Model size mismatch: {path}: expected {item.bytes}, got {stat.st_size}; run models download to repair")
        if sha256_file(path) != item.sha256:
            raise ValueError(f"Model SHA-256 mismatch: {path}; run models download to repair")
        self._verified[item.name] = signature
        logger.debug("Verified model %s sha256=%s", path, item.sha256)
        return path

    def check(self, mode="all", *, force=False):
        return {item.name: self.verify(item, force=force) for item in self.selected(mode)}

    def report(self, mode="all"):
        """Report every missing/corrupt file, rather than stopping at the first."""
        results = []
        for item in self.selected(mode):
            try:
                self.verify(item, force=True)
                results.append({"name": item.name, "status": "ok"})
            except (OSError, ValueError) as error:
                results.append({"name": item.name, "status": "error", "error": str(error)})
        return results

    @contextmanager
    def _download_lock(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".download.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def download(self, mode="all"):
        with self._download_lock():
            for item in self.selected(mode):
                try:
                    self.verify(item)
                    logger.info("Already verified: %s", item.name)
                    continue
                except (OSError, ValueError):
                    pass
                destination = self.directory / item.name
                fd, name = tempfile.mkstemp(prefix=f".{item.name}.", suffix=".part", dir=self.directory)
                temporary = Path(name)
                try:
                    with os.fdopen(fd, "wb") as target:
                        if item.bundled:
                            with files("flashvsr").joinpath("assets", item.name).open("rb") as source:
                                shutil.copyfileobj(source, target)
                        else:
                            url = (f"https://huggingface.co/{self.release['repository']}/resolve/"
                                   f"{self.release['revision']}/{item.name}")
                            logger.info("Downloading %s (%s bytes)", item.name, item.bytes)
                            with urllib.request.urlopen(url, timeout=60) as source:
                                total = 0
                                while block := source.read(4 * 1024 * 1024):
                                    total += len(block)
                                    if total > item.bytes:
                                        raise ValueError(f"Download exceeds manifest size: {item.name}")
                                    target.write(block)
                        target.flush()
                        os.fsync(target.fileno())
                    self.verify(item, temporary, force=True)
                    os.replace(temporary, destination)
                    logger.info("Downloaded and verified: %s", destination)
                finally:
                    temporary.unlink(missing_ok=True)
        return self.check(mode)

    def identity(self, mode):
        return {"version": self.version, "repository": self.release["repository"],
                "revision": self.release["revision"], "directory": str(self.directory),
                "files": {item.name: item.sha256 for item in self.selected(mode)}}
