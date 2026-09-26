"""Preflight checks shared by installation, inference and GPU acceptance runs."""

import os
from pathlib import Path
import re
import subprocess
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]


def validate_python():
    if not sys.platform.startswith("linux"):
        raise RuntimeError("FlashVSR-Pro supports native Linux and Google Colab only")
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("Use Python 3.10–3.12 with the pinned dependencies")


def validate_toolkit(cuda_home):
    """Require the same toolkit minor version as the cu124 wheel baseline."""
    validate_python()
    compiler = Path(cuda_home) / "bin/nvcc"
    if not compiler.is_file():
        raise RuntimeError(f"CUDA compiler not found: {compiler}")
    version = subprocess.check_output([str(compiler), "--version"], text=True)
    match = re.search(r"release (\d+)\.(\d+)", version)
    if not match or tuple(map(int, match.groups())) != (12, 4):
        raise RuntimeError("Use CUDA Toolkit 12.4 with PyTorch 2.6.0+cu124; set CUDA_HOME to that toolkit")
    return version.strip()


def model_directory():
    return Path(os.getenv(
        "FLASHVSR_MODEL_PATH",
        os.getenv("FLASHVSR-Pro_MODEL_PATH", str(PROJECT_DIR / "models/FlashVSR-v1.1")),
    )).expanduser().resolve()


def require_weight(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Model weight missing or empty: {path}. See INSTALLATION.md.")
    with path.open("rb") as source:
        if source.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"Model file is a Git LFS pointer, not downloaded weights: {path}")
    return path


def validate_models(mode, model_dir=None, project_dir=PROJECT_DIR):
    if mode not in ("full", "tiny", "tiny-long"):
        raise ValueError(f"Unsupported inference mode: {mode}")
    directory = model_directory() if model_dir is None else Path(model_dir)
    names = (
        "diffusion_pytorch_model_streaming_dmd.safetensors",
        "LQ_proj_in.ckpt",
        "Wan2.1_VAE.pth" if mode == "full" else "TCDecoder.ckpt",
    )
    for name in names:
        require_weight(directory / name)
    require_weight(Path(project_dir) / "models/prompt_tensor/posi_prompt.pth")
    return directory


def validate_cuda(torch, device="cuda", dtype="bf16"):
    validate_python()
    if not re.fullmatch(r"cuda(?::\d+)?", str(device)):
        raise ValueError("The sparse backend requires cuda or cuda:N; CPU inference is unsupported")
    if dtype not in ("fp16", "bf16"):
        raise ValueError("The sparse backend requires fp16 or bf16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver or select a Colab GPU runtime.")
    if torch.__version__.split("+")[0] != "2.6.0" or torch.version.cuda != "12.4":
        raise RuntimeError("This baseline requires PyTorch 2.6.0+cu124 with CUDA 12.4; run scripts/install.sh")
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
