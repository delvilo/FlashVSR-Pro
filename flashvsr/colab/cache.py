"""Versioned CUDA extension wheels, reusable only after identity and GPU checks."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def source_identity(project, directory="Block-Sparse-Attention"):
    project = Path(project)
    paths = subprocess.check_output(["git", "ls-files", "-z", "--", directory], cwd=project).decode().split("\0")
    digest = hashlib.sha256()
    for name in sorted(filter(None, paths)):
        path = project / name
        digest.update(name.encode() + b"\0")
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    if not any(paths):
        raise RuntimeError("A Git checkout with the bundled attention sources is required")
    return digest.hexdigest()


def cache_key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class WheelCache:
    def __init__(self, store, identity):
        self.store, self.identity = store, identity
        self.key = cache_key(identity)
        self.directory = Path("wheels") / self.key

    def restore(self, destination):
        record = self.store.path(self.directory / "manifest.json")
        if not record.is_file():
            return None
        try:
            manifest = json.loads(record.read_text())
            if manifest["identity"] != self.identity:
                return None
            name = manifest["wheel"]
            if Path(name).name != name or not name.endswith(".whl"):
                return None
            expected = manifest["file"]
            target = Path(destination) / name
            self.store.restore(self.directory / name, target, expected)
            return target
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def publish(self, wheel):
        wheel = Path(wheel)
        identity = self.store.copy(wheel, self.directory / wheel.name)
        self.store.write_json(self.directory / "manifest.json",
                              {"schema": 1, "identity": self.identity, "wheel": wheel.name, "file": identity})


def ensure_wheel(cache, local_root, build, install, smoke):
    """A cached wheel that fails a real CUDA operation gets one source rebuild."""
    local_root = Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wheel-", dir=local_root) as temp:
        wheel = cache.restore(temp)
        if wheel:
            try:
                install(wheel)
                smoke()
                return {"key": cache.key, "status": "reused"}
            except (OSError, RuntimeError, subprocess.SubprocessError):
                # Do not let the rejected archive satisfy a source build's
                # output lookup, even if the new build changes its filename.
                wheel.unlink(missing_ok=True)
        wheel = Path(build(Path(temp)))
        install(wheel)
        smoke()
        # An import alone is insufficient: publish only after CUDA execution.
        cache.publish(wheel)
        return {"key": cache.key, "status": "built"}
