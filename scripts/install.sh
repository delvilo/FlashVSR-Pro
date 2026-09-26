#!/usr/bin/env bash
# Install into the selected Python environment on Linux or Google Colab.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash scripts/install.sh

Linux: activate a Python 3.10-3.12 virtual environment first.
Colab: select an Ampere/Ada/Hopper GPU runtime and use the notebook instructions.
Install an NVIDIA driver, CUDA Toolkit 12.4+, a C++ compiler, FFmpeg and Git first.
The pinned PyTorch build uses CUDA 12.4; a matching toolkit is recommended.

Environment overrides:
  FLASHVSR_PYTHON               Python executable (default: python)
  CUDA_HOME                    CUDA Toolkit directory (otherwise detected from nvcc)
  BLOCK_SPARSE_ATTN_CUDA_ARCHS  Kernel targets (default: 80;90)
  MAX_JOBS                     Parallel build jobs (default: 2)
  NVCC_THREADS                 Threads per compiler process (default: 2)

Model weights are downloaded separately; see INSTALLATION.md.
HELP
    exit 0
fi
if (( $# != 0 )); then
    echo "Unknown argument. Run bash scripts/install.sh --help." >&2
    exit 2
fi

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_bin="${FLASHVSR_PYTHON:-python}"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This installer targets Linux and Google Colab. See INSTALLATION.md." >&2
    exit 1
fi
for executable in "$python_bin" git g++ ffmpeg ffprobe; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        echo "Required executable not found: $executable. See INSTALLATION.md." >&2
        exit 1
    fi
done

"$python_bin" - <<'PY'
import sys
if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    raise SystemExit("The pinned dependencies require Python 3.10-3.12.")
PY

if [[ -z "${CUDA_HOME:-}" ]]; then
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "nvcc not found. Install CUDA Toolkit 12.4+ and set CUDA_HOME." >&2
        exit 1
    fi
    CUDA_HOME="$(dirname -- "$(dirname -- "$(readlink -f -- "$(command -v nvcc)")")")"
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
"$python_bin" - <<'PY'
import os
import re
import subprocess
from pathlib import Path
nvcc = Path(os.environ["CUDA_HOME"]) / "bin/nvcc"
if not nvcc.is_file():
    raise SystemExit(f"CUDA compiler not found: {nvcc}")
version = subprocess.check_output([str(nvcc), "--version"], text=True)
match = re.search(r"release (\d+)\.(\d+)", version)
if not match or not (12, 4) <= tuple(map(int, match.groups())) < (13, 0):
    raise SystemExit("Use CUDA Toolkit 12.4+ (12.x) with the pinned PyTorch cu124 build.")
print(version.strip())
PY

"$python_bin" -m pip install --upgrade pip setuptools wheel packaging ninja psutil
"$python_bin" -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Fail before compiling the extension if the selected GPU cannot run its kernels.
"$python_bin" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Check the NVIDIA driver or select a Colab GPU runtime.")
major, minor = torch.cuda.get_device_capability()
if major not in (8, 9):
    raise SystemExit("This dependency baseline supports Ampere/Ada/Hopper GPUs. "
                     "Colab T4/P100 GPUs are not supported; choose A100 or L4.")
print(f"GPU: {torch.cuda.get_device_name()} (compute capability {major}.{minor})")
PY

"$python_bin" -m pip install -r requirements.txt
"$python_bin" -m pip install --no-deps -e .

# CUDA 12.4 cannot compile the backend's default sm_120 target.
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="${BLOCK_SPARSE_ATTN_CUDA_ARCHS:-80;90}"
export MAX_JOBS="${MAX_JOBS:-2}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
BLOCK_SPARSE_ATTN_FORCE_BUILD=TRUE "$python_bin" -m pip install \
    --no-build-isolation --no-deps ./Block-Sparse-Attention

mkdir -p inputs results
"$python_bin" -c 'import block_sparse_attn; import diffsynth; print("FlashVSR-Pro imports OK")'
"$python_bin" infer.py --help
echo "Installation complete. Download model weights as described in INSTALLATION.md."
