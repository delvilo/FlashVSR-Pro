"""Prepare only the Ubuntu 24.04 / Python 3.13 / cu128 Colab baseline."""

import hashlib
import importlib.metadata
import logging
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import time

from flashvsr.media import MediaTools
from flashvsr.observability import write_report
from utils.runtime import validate_cuda, validate_toolkit
from .cache import WheelCache, ensure_wheel, source_identity
from .storage import GB, local_lock, prepare_models, require_local_space

logger = logging.getLogger(__name__)


def output(command, **kwargs):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30, **kwargs).strip()


def validate_baseline(torch, release=None, python_version=None):
    release = platform.freedesktop_os_release() if release is None else release
    python_version = sys.version_info[:2] if python_version is None else python_version
    if release.get("ID") != "ubuntu" or release.get("VERSION_ID") != "24.04" or release.get("VERSION_CODENAME") != "noble":
        raise RuntimeError("Colab requires Ubuntu 24.04 noble; older Colab runtimes are unsupported")
    if tuple(python_version) != (3, 13):
        raise RuntimeError("Colab requires its native Python 3.13 runtime")
    index = validate_cuda(torch)
    name = torch.cuda.get_device_name(index)
    if "A100" not in name and not re.search(r"\bL4\b", name):
        raise RuntimeError("Select an A100 or L4 Colab runtime; T4/P100 are unsupported")
    return index


def toolkit_directory():
    explicit = os.getenv("CUDA_HOME")
    if explicit:
        return Path(explicit).resolve()
    pinned = Path("/usr/local/cuda-12.8")
    if (pinned / "bin/nvcc").is_file():
        return pinned
    compiler = shutil.which("nvcc")
    if not compiler:
        raise RuntimeError("CUDA Toolkit 12.8 nvcc is missing; use the current Colab runtime")
    return Path(compiler).resolve().parent.parent


def diagnose(project, store, local_root, *, check_encoder=True):
    import torch
    index = validate_baseline(torch)
    cuda_home = toolkit_directory()
    compiler = validate_toolkit(cuda_home)
    if not re.search(r"release 12\.8(?:,|\s)", compiler):
        raise RuntimeError("The Colab baseline uses CUDA Toolkit 12.8; set CUDA_HOME to its directory")
    headers = Path(sysconfig.get_path("include"))
    if not (headers / "Python.h").is_file() or not (headers / "patchlevel.h").is_file():
        raise RuntimeError(f"Python 3.13 headers missing at {headers}; use the native Colab Python, not Ubuntu's python3-dev")
    patchlevel = (headers / "patchlevel.h").read_text()
    if not re.search(r"#define\s+PY_MINOR_VERSION\s+13\b", patchlevel):
        raise RuntimeError("Python development headers do not match Python 3.13")
    cxx = shutil.which("g++")
    if not cxx or not shutil.which("gcc"):
        raise RuntimeError("gcc/g++ is missing; install build-essential before setup")
    media = MediaTools()
    data = {"os": platform.freedesktop_os_release(), "python": platform.python_version(),
            "executable": sys.executable, "python_abi": sysconfig.get_config_var("SOABI"),
            "python_headers": str(headers), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(index), "compute_capability": list(torch.cuda.get_device_capability(index)),
            "vram_bytes": torch.cuda.get_device_properties(index).total_memory,
            "torch_architectures": torch.cuda.get_arch_list(),
            "cuda_home": str(cuda_home), "nvcc": compiler, "cxx": output([cxx, "--version"]),
            "nvidia_smi": output(["nvidia-smi"]), "ffmpeg": media.ffmpeg_version,
            "ffprobe": output([media.ffprobe, "-version"]).splitlines()[0],
            "local_free_bytes": shutil.disk_usage(local_root).free, "drive": store.capacity(),
            "code_commit": output(["git", "rev-parse", "HEAD"], cwd=project)}
    if check_encoder:
        data["hevc_nvenc_test"] = media.nvenc_works(index)
        data["encoder"] = "hevc_nvenc" if data["hevc_nvenc_test"] else "libx264"
    data["wheel_identity"] = {"schema": 1, "python_abi": data["python_abi"],
                              "machine": platform.machine(), "libc": list(platform.libc_ver()),
                              "torch": data["torch"], "cuda": data["torch_cuda"],
                              "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                              "gpu_capability": data["compute_capability"], "nvcc": compiler,
                              "cxx": data["cxx"], "architectures": "80",
                              "build_requirements": hashlib.sha256((Path(project) / "requirements-build.txt").read_bytes()).hexdigest(),
                              "source_sha256": source_identity(project)}
    return data


def verify_requirements(project):
    """Validate the application's dependency closure, not unrelated Colab apps."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    queue = [line.strip() for line in (Path(project) / "requirements.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    seen, versions = set(), {}
    while queue:
        requirement = Requirement(queue.pop())
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        name = canonicalize_name(requirement.name)
        distribution = importlib.metadata.distribution(name)
        if not requirement.specifier.contains(distribution.version, prereleases=True):
            raise RuntimeError(f"Dependency mismatch: {requirement}; installed {distribution.version}")
        versions[name] = distribution.version
        if name not in seen:
            seen.add(name)
            queue.extend(distribution.requires or [])
    return versions


def cuda_smoke():
    import torch
    from block_sparse_attn import block_sparse_attn_func
    validate_baseline(torch)
    q = torch.randn(256, 2, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    k, v = torch.randn_like(q) * 0.1, torch.randn_like(q)
    lengths = torch.tensor([0, 256], device="cuda", dtype=torch.int32)
    heads = torch.ones(2, device="cuda", dtype=torch.int32)
    mask = torch.ones((1, 2, 2, 2), device="cuda", dtype=torch.bool)
    actual = block_sparse_attn_func(q, k, v, lengths, lengths, heads, None, mask,
                                    256, 256, 0.0, is_causal=False)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0))
    torch.testing.assert_close(actual, expected.squeeze(0).transpose(0, 1), rtol=0.04, atol=0.01)
    torch.cuda.synchronize()


def prepare(project, store, local_root, mode="tiny"):
    project, local_root = Path(project).resolve(), Path(local_root).resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    log = local_root / "setup.log"
    report = {"status": "running", "mode": mode}
    started = time.perf_counter()
    with local_lock(local_root / ".colab.lock"):
        store.recover_partials()
        try:
            # Validate OS/Python/GPU/torch/toolkit before touching packages.
            report["diagnostics"] = diagnose(project, store, local_root)
            env = dict(os.environ, CUDA_HOME=report["diagnostics"]["cuda_home"],
                       BLOCK_SPARSE_ATTN_CUDA_ARCHS="80", MAX_JOBS="2", NVCC_THREADS="2",
                       BLOCK_SPARSE_ATTN_FORCE_BUILD="TRUE", PIP_DISABLE_PIP_VERSION_CHECK="1",
                       CXX=shutil.which("g++"), CC=shutil.which("gcc"))
            env["PATH"] = str(Path(env["CUDA_HOME"]) / "bin") + os.pathsep + env["PATH"]
            # Large torch/cu128 wheels already belong to Colab. Constraints make
            # accidental replacement a resolver error rather than a CUDA change.
            constraint = local_root / "colab-constraints.txt"
            constraint.write_text("torch==2.11.0+cu128\n")

            def run(command):
                logger.info("Setup: %s", " ".join(map(str, command)))
                with log.open("a") as stream:
                    result = subprocess.run(list(map(str, command)), cwd=project, env=env,
                                            stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"Setup command failed ({result.returncode}); see {log}\n" + log.read_text(errors="replace")[-3000:])

            log.write_text("")
            pip = [sys.executable, "-m", "pip"]
            run([*pip, "install", "-c", constraint, "-r", "requirements-build.txt"])
            run([*pip, "install", "-c", constraint, "-r", "requirements.txt"])
            run([*pip, "install", "--no-build-isolation", "--no-deps", "-e", "."])
            run([sys.executable, "-m", "flashvsr.colab", "check-dependencies"])
            cache = WheelCache(store, report["diagnostics"]["wheel_identity"])

            def build(directory):
                require_local_space(local_root, 3 * GB)
                run([*pip, "wheel", "--no-build-isolation", "--no-deps", "--no-cache-dir",
                     "--wheel-dir", directory, "./Block-Sparse-Attention"])
                wheels = list(directory.glob("block_sparse_attn-*.whl"))
                if len(wheels) != 1:
                    raise RuntimeError("Expected exactly one compiled attention wheel")
                return wheels[0]

            report["wheel"] = ensure_wheel(cache, local_root / "builds", build,
                lambda wheel: run([*pip, "install", "--no-deps", "--force-reinstall", wheel]),
                lambda: run([sys.executable, "-m", "flashvsr.colab", "smoke"]))
            report["model_directory"] = str(prepare_models(store, local_root, mode))
            report["packages"] = verify_requirements(project)
            report["status"] = "ok"
        except BaseException as error:
            report.update(status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
            raise
        finally:
            report["seconds"] = time.perf_counter() - started
            try:
                report["drive"] = store.capacity()
            except (OSError, ValueError) as error:
                report["drive_error"] = str(error)
            write_report(local_root / "setup.json", report)
            try:
                store.copy(local_root / "setup.json", "diagnostics/setup.json")
                if log.is_file():
                    store.copy(log, "diagnostics/setup.log")
            except (OSError, ValueError):
                if report["status"] == "ok":
                    raise
                logger.exception("Could not persist setup failure report; local copy: %s", local_root)
    return report
