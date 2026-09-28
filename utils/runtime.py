"""Preflight checks shared by installation, inference and GPU acceptance runs."""

from pathlib import Path
import re
import subprocess
import sys

from flashvsr.models import model_directory

PROJECT_DIR = Path(__file__).resolve().parents[1]


def validate_python():
    if not sys.platform.startswith("linux"):
        raise RuntimeError("FlashVSR-Pro supports native Linux and Google Colab only")
    if not (3, 13) <= sys.version_info[:2] < (3, 15):
        raise RuntimeError("Use Python 3.13–3.14 with the pinned CUDA wheels")


def validate_toolkit(cuda_home):
    """CUDA 12.8+ (within 12.x) compilers can build the cu128 extension."""
    validate_python()
    compiler = Path(cuda_home) / "bin/nvcc"
    if not compiler.is_file():
        raise RuntimeError(f"CUDA compiler not found: {compiler}")
    version = subprocess.check_output([str(compiler), "--version"], text=True)
    match = re.search(r"release (\d+)\.(\d+)", version)
    if not match or not (12, 8) <= tuple(map(int, match.groups())) < (13, 0):
        raise RuntimeError("Use CUDA Toolkit 12.8 or newer (12.x) with PyTorch 2.11.0+cu128; set CUDA_HOME to that toolkit")
    return version.strip()


def require_weight(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Model weight missing or empty: {path}. See INSTALLATION.md.")
    with path.open("rb") as source:
        if source.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Model file is a Git LFS pointer, not downloaded weights: {path}")
    return path


def validate_models(mode, model_dir=None):
    from flashvsr.models import ModelRegistry
    registry = ModelRegistry(directory=model_dir)
    registry.check(mode)
    return registry.directory


def validate_cuda(torch, device="cuda", dtype="bf16"):
    validate_python()
    if not re.fullmatch(r"cuda(?::\d+)?", str(device)):
        raise ValueError("The sparse backend requires cuda or cuda:N; CPU inference is unsupported")
    if dtype not in ("fp16", "bf16"):
        raise ValueError("The sparse backend requires fp16 or bf16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver or select a Colab GPU runtime.")
    if torch.__version__ != "2.11.0+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError("This baseline requires PyTorch 2.11.0+cu128; run scripts/install.sh")
    index = int(device.split(":")[1]) if ":" in device else torch.cuda.current_device()
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError(f"CUDA device index is unavailable: {device}")
    major, minor = torch.cuda.get_device_capability(index)
    if major not in (8, 9):
        raise RuntimeError("Use an Ampere/Ada/Hopper GPU (Colab A100 or L4); T4/P100 are unsupported")
    torch.cuda.set_device(index)
    if dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support BF16; use --dtype fp16")
    return index
