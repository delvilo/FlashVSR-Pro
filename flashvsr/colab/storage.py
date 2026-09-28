"""Budgeted, verified file copies for a mounted Drive (one active notebook)."""

from contextlib import contextmanager
import errno
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import shutil
import tempfile

from flashvsr.jobs import file_identity

logger = logging.getLogger(__name__)
GB = 1_000_000_000


def owned_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or any(p in ("..", ".") for p in relative.parts):
        raise ValueError(f"Invalid managed path: {relative}")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlinks are not allowed in managed storage: {path}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes managed storage: {path}")
    return path


def sync_file(stream):
    stream.flush()
    try:
        os.fsync(stream.fileno())
    except OSError as error:
        # Some Drive FUSE versions do not implement fsync. Read-back validation
        # still applies, but the mount cannot promise server-side durability.
        if error.errno not in (errno.EINVAL, errno.ENOTSUP, errno.ENOSYS):
            raise


def require_local_space(path, additional, reserve=GB):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path).free < additional + reserve:
        raise OSError(f"Insufficient local disk space at {path}; need {additional + reserve:,} free bytes")


def copy_verified(source, destination, identity=None, *, reuse_existing=True):
    """Replace only after a complete copy has passed SHA-256 read-back."""
    source, destination = Path(source), Path(destination)
    identity = identity or file_identity(source)
    if destination.is_symlink():
        raise ValueError(f"Refusing to replace a symlink: {destination}")
    if reuse_existing and destination.is_file() and file_identity(destination) == identity:
        return identity
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".flashvsr-", suffix=".part", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as writer, source.open("rb") as reader:
            shutil.copyfileobj(reader, writer, length=4 * 1024 * 1024)
            sync_file(writer)
        if file_identity(temporary) != identity:
            raise ValueError(f"Copy verification failed: {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return identity


@contextmanager
def local_lock(path):
    """Serialize setup/run cells in this VM; Drive is not a distributed lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another Colab setup/run is active in this runtime") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class DriveStore:
    def __init__(self, root, budget_gb=20, reserve_gb=2):
        if not all(math.isfinite(v) for v in (budget_gb, reserve_gb)) or not 0 <= reserve_gb < budget_gb:
            raise ValueError("Require 0 <= reserve GB < Drive budget GB")
        self.root = Path(root).expanduser().resolve()
        self.budget = int(budget_gb * GB)
        self.reserve = int(reserve_gb * GB)
        self.root.mkdir(parents=True, exist_ok=True)
        self._verified = {}

    def path(self, relative):
        return owned_path(self.root, relative)

    def usage(self):
        total = 0
        for path in self.root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Symlink found in managed Drive directory: {path}")
            if path.is_file():
                total += path.stat().st_size
        return total

    def capacity(self):
        return {"root": str(self.root), "budget_bytes": self.budget,
                "reserve_bytes": self.reserve, "used_bytes": self.usage(),
                "mount_free_bytes": shutil.disk_usage(self.root).free}

    def require_space(self, additional):
        # Count the entire temporary copy before replacing an existing file.
        # The mount's free count is NOT a reliable Google account quota API.
        usage = self.usage()
        if usage + additional + self.reserve > self.budget or shutil.disk_usage(self.root).free < additional + self.reserve:
            raise OSError("Drive budget/free space is insufficient; completed checkpoints are retained. "
                          f"Managed usage {usage / GB:.2f} GB, additional {additional / GB:.2f} GB, "
                          f"budget {self.budget / GB:g} GB, reserve {self.reserve / GB:g} GB. "
                          "Free space or raise the budget, then resume the same job.")

    def copy(self, source, relative, identity=None):
        source = Path(source)
        identity = identity or file_identity(source)
        destination = self.path(relative)
        if destination.is_file():
            stat = destination.stat()
            signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, identity["sha256"])
            if self._verified.get(str(relative)) == signature or file_identity(destination) == identity:
                self._verified[str(relative)] = signature
                return identity
        self.require_space(identity["bytes"])
        copy_verified(source, destination, identity)
        stat = destination.stat()
        self._verified[str(relative)] = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, identity["sha256"])
        return identity

    def write_json(self, relative, data):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "record.json"
            source.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            return self.copy(source, relative)

    def restore(self, relative, destination, identity):
        """Read Drive once, verify the local copy, then remember the source stat."""
        source = self.path(relative)
        before = source.stat()
        if before.st_size != identity["bytes"]:
            raise ValueError(f"Cached file size mismatch: {relative}")
        require_local_space(Path(destination).parent, identity["bytes"])
        copy_verified(source, destination, identity, reuse_existing=False)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(f"Cache changed while copying: {relative}")
        self._verified[str(relative)] = (after.st_size, after.st_mtime_ns, after.st_ctime_ns, identity["sha256"])

    def recover_partials(self):
        # Only this helper's uncommitted files are eligible; no user outputs.
        for path in self.root.rglob(".flashvsr-*.part"):
            path = self.path(path.relative_to(self.root))
            path.unlink()


def prepare_models(store, local_root, mode):
    """Download only missing weights locally, then verify both cache copies."""
    from flashvsr.models import ModelRegistry
    local_root = Path(local_root)
    registry = ModelRegistry(directory=local_root / "models/v1.1")
    registry.directory.mkdir(parents=True, exist_ok=True)
    for item in registry.selected(mode):
        relative = Path("models/v1.1") / item.name
        local = registry.directory / item.name
        expected = {"bytes": item.bytes, "sha256": item.sha256}
        try:
            registry.verify(item, local)
        except (OSError, ValueError):
            try:
                store.restore(relative, local, expected)
            except (FileNotFoundError, ValueError):
                store.require_space(item.bytes)
                require_local_space(registry.directory, item.bytes)
                # Use the existing pinned downloader and its atomic SHA check.
                selected = ModelRegistry(directory=registry.directory)
                selected.entries = (item,)
                selected.download(mode)
        store.copy(local, relative, expected)
    return registry.directory
