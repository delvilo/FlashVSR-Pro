#!/usr/bin/env bash
# Install into the selected Python environment on Linux or Google Colab.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash scripts/install.sh

Linux: activate a Python 3.10-3.12 virtual environment first.
Colab: select an Ampere/Ada/Hopper GPU runtime and use the notebook instructions.
Install an NVIDIA driver, CUDA Toolkit 12.4, a C++ compiler, FFmpeg and Git first.
The supported build uses PyTorch 2.6.0+cu124 and CUDA Toolkit 12.4.

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
for executable in "$python_bin" git g++ "${FLASHVSR_FFMPEG:-ffmpeg}" "${FLASHVSR_FFPROBE:-ffprobe}"; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        echo "Required executable not found: $executable. See INSTALLATION.md." >&2
        exit 1
    fi
done

"$python_bin" - <<'PY'
from utils.runtime import validate_python
try:
    validate_python()
except RuntimeError as error:
    raise SystemExit(str(error))
PY

if [[ -z "${CUDA_HOME:-}" ]]; then
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "nvcc not found. Install CUDA Toolkit 12.4 and set CUDA_HOME." >&2
        exit 1
    fi
    CUDA_HOME="$(dirname -- "$(dirname -- "$(readlink -f -- "$(command -v nvcc)")")")"
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
"$python_bin" - <<'PY'
import os
from utils.runtime import validate_toolkit
try:
    print(validate_toolkit(os.environ["CUDA_HOME"]))
except (RuntimeError, OSError) as error:
    raise SystemExit(str(error))
PY

"$python_bin" -m pip install -r requirements-build.txt
"$python_bin" -m pip install -r requirements-cuda.txt

# Fail before compiling the extension if the selected GPU cannot run its kernels.
"$python_bin" - <<'PY'
import torch
from utils.runtime import validate_cuda
try:
    validate_cuda(torch)
except (RuntimeError, ValueError) as error:
    raise SystemExit(str(error))
print(f"GPU: {torch.cuda.get_device_name()}")
PY

# OpenCV distributions share cv2; remove conflicting variants before installing
# the single headless build used by native Linux and Colab.
"$python_bin" -m pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless opencv-python-headless
"$python_bin" -m pip install -r requirements.txt
"$python_bin" -m pip install --no-build-isolation --no-deps -e .

# CUDA 12.4 cannot compile the backend's default sm_120 target.
export BLOCK_SPARSE_ATTN_CUDA_ARCHS="${BLOCK_SPARSE_ATTN_CUDA_ARCHS:-80;90}"
export MAX_JOBS="${MAX_JOBS:-2}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
BLOCK_SPARSE_ATTN_FORCE_BUILD=TRUE "$python_bin" -m pip install \
    --no-build-isolation --no-deps ./Block-Sparse-Attention

mkdir -p inputs results
"$python_bin" -m pip check
"$python_bin" -c 'import block_sparse_attn; import diffsynth; print("FlashVSR-Pro imports OK")'
"$python_bin" infer.py --help
"$python_bin" - <<'REPORT'
import importlib.metadata
import json
import platform
from pathlib import Path
import torch
report = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
          "packages": sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())}
Path("results/environment.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
REPORT
echo "Installation complete. Download model weights as described in INSTALLATION.md."
