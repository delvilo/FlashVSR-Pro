# FlashVSR-Pro installation

This guide covers direct installation and execution on **native Linux and Google Colab**. Windows, macOS, and WSL 2 are not supported.

## Requirements

- Linux with an NVIDIA driver and an Ampere, Ada, or Hopper GPU.
- Python 3.10–3.12. Python 3.11 is the Linux baseline.
- CUDA Toolkit **12.4 recommended**, including `nvcc`. The installer accepts 12.4+ toolkits within CUDA 12.x; the pinned PyTorch packages use CUDA 12.4.
- A C++17 compiler, Git, FFmpeg, and FFprobe.
- Enough disk space for model weights, build files, input videos, and results.
- VRAM requirements vary with resolution and duration. Begin with a short clip and `--tile-dit`.

The bundled attention backend cannot run on Colab T4/P100 GPUs. The pinned setup targets Ampere/Ada/Hopper; newer GPU generations require a separate dependency upgrade.

## Linux

### 1. System packages

For Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git git-lfs ffmpeg python3-venv python3-dev
```

Install the NVIDIA driver and CUDA Toolkit using [NVIDIA's Linux installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). Select CUDA 12.4 for the closest match to the pinned PyTorch build. Verify:

```bash
nvidia-smi
nvcc --version
```

The CUDA version displayed by `nvidia-smi` describes driver compatibility; `nvcc --version` identifies the installed compiler. A working driver alone is insufficient to build the extension.

If CUDA is outside `PATH`, set its actual location, for example:

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
```

### 2. Checkout and Python environment

```bash
git clone --depth 1 https://github.com/delvilo/FlashVSR-Pro.git
cd FlashVSR-Pro
python3 --version
python3 -m venv .venv
source .venv/bin/activate
```

If the system Python is outside 3.10–3.12, install Python 3.11 with its matching development and venv packages, then create the environment using `python3.11 -m venv .venv`. You can instead activate an existing Conda environment with a supported Python version.

The sparse attention source and CUTLASS headers are included as regular files in this checkout. No submodule initialization is required. See [THIRD_PARTY.md](THIRD_PARTY.md) for dependency sources and licenses. The shallow clone avoids downloading historical sample binaries and generated documentation; existing Git history is unchanged.

### 3. Install

```bash
bash scripts/install.sh
```

The installer uses the active `python` interpreter, checks prerequisites, installs PyTorch 2.6.0/torchvision 0.21.0/torchaudio 2.6.0 from the CUDA 12.4 index, installs the application dependencies, and builds the bundled CUDA extension against that PyTorch installation. It creates `inputs/` and `results/` and checks application imports.

To select an explicit interpreter:

```bash
FLASHVSR_PYTHON=/path/to/venv/bin/python bash scripts/install.sh
```

Build settings can be overridden:

```bash
MAX_JOBS=2 NVCC_THREADS=2 BLOCK_SPARSE_ATTN_CUDA_ARCHS='80;90' bash scripts/install.sh
```

The defaults limit compiler memory usage and select architectures supported by the CUDA 12.4 baseline. Do not select `100`, `110`, or `120` for a CUDA 12.4 compiler.

### 4. Download weights

From the project directory and the activated Python environment:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('JunhaoZhuang/FlashVSR-v1.1', local_dir='models/FlashVSR-v1.1')"
```

This downloads actual model data. Downloading a Git repository without its LFS objects would leave pointer files that cannot be loaded as weights.

| File | Used by |
| --- | --- |
| `models/FlashVSR-v1.1/diffusion_pytorch_model_streaming_dmd.safetensors` | All modes |
| `models/FlashVSR-v1.1/LQ_proj_in.ckpt` | All modes |
| `models/FlashVSR-v1.1/TCDecoder.ckpt` | `tiny`, `tiny-long` |
| `models/FlashVSR-v1.1/Wan2.1_VAE.pth` | `full` |
| `models/prompt_tensor/posi_prompt.pth` | Bundled in the checkout; all modes |

All required weights should be downloaded before inference. Decoder loading does not automatically fetch missing files.

To use another location, download into that directory and set `FLASHVSR_MODEL_PATH`. Both DiT/projector and decoder loading use it. User-supplied relative paths resolve from the current directory; the default weights and bundled prompt resolve from the project location.

### 5. Verify and run

```bash
python -c "import torch; import block_sparse_attn; import diffsynth; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
python infer.py --help
python scripts/download_samples.py example0.mp4
python infer.py -i inputs/example0.mp4 -o results/ --mode tiny --scale 4.0 --tile-dit
```

The sample downloader needs only Python's standard library. It fetches the selected video from a fixed repository commit and verifies its byte count and SHA-256 before saving it. Use your own video path to skip the sample download. See [inputs/README.md](inputs/README.md) for the optional collection.

For subsequent sessions, activate the same environment. You may call the scripts using absolute paths from another directory; the bundled models and prompt do not depend on the shell's working directory.

## Google Colab

The ready-to-run notebook is [colab/FlashVSR_Pro.ipynb](colab/FlashVSR_Pro.ipynb). Open it in Colab and run its cells in order.

### Runtime selection

Choose **Runtime → Change runtime type → GPU**, with an **A100 or L4** when available. GPU availability varies. T4/P100 cannot run the sparse attention kernels; switching Python packages will not solve that hardware limitation.

The notebook uses its own `sys.executable` for installation and every inference subprocess. No Conda activation is needed. It avoids importing the inference stack into the notebook kernel, so installing the pinned packages does not leave an already-imported PyTorch version in the inference process.

The runtime needs Python 3.10–3.12 and a CUDA 12.4+ toolkit in the 12.x series. If a runtime's Python or toolkit is outside that range, use a compatible runtime or install the baseline environment before proceeding. Check toolkit availability with `nvcc --version`; do not substitute the driver version from `nvidia-smi`.

### Manual cells

First clone the code and install system packages:

```python
from pathlib import Path
import subprocess
import sys

project = Path('/content/FlashVSR-Pro')
subprocess.run(['apt-get', 'update'], check=True)
subprocess.run(['apt-get', 'install', '-y', 'build-essential', 'git', 'git-lfs', 'ffmpeg', 'python3-dev'], check=True)
if not project.exists():
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/delvilo/FlashVSR-Pro.git', str(project)], check=True)
```

Install into the notebook's Python environment:

```python
import os

install_env = os.environ.copy()
install_env['FLASHVSR_PYTHON'] = sys.executable
# If needed, select an installed compatible toolkit:
# install_env['CUDA_HOME'] = '/usr/local/cuda-12.4'
subprocess.run(['bash', 'scripts/install.sh'], cwd=project, env=install_env, check=True)
```

Download models and choose an input. The following cell uses an optional sample; replace `input_video` with an uploaded video or a path on mounted Google Drive to use your own:

```python
subprocess.run([
    sys.executable, '-c',
    "from huggingface_hub import snapshot_download; "
    "snapshot_download('JunhaoZhuang/FlashVSR-v1.1', local_dir='models/FlashVSR-v1.1')"
], cwd=project, check=True)

input_video = project / 'inputs/example0.mp4'  # Or Path('/content/input.mp4').
output_video = Path('/content/enhanced.mp4')
if input_video == project / 'inputs/example0.mp4':
    subprocess.run([
        sys.executable, str(project / 'scripts/download_samples.py'), 'example0.mp4'
    ], cwd=project, check=True)
if not input_video.is_file():
    raise FileNotFoundError(input_video)
subprocess.run([
    sys.executable, str(project / 'infer.py'),
    '-i', str(input_video), '-o', str(output_video),
    '--mode', 'tiny', '--scale', '4.0', '--tile-dit', '--keep-audio'
], cwd=project, check=True)
```

Download the output before the runtime ends:

```python
from google.colab import files
files.download(str(output_video))
```

Colab runtime disks are temporary. For large files, mount Google Drive and set the input/output paths there. After a runtime reset, repeat installation or restore the setup in the new runtime.

## Manual dependency installation

The installer is preferred because it validates the selected environment. The equivalent installation order, from the checkout in a supported Python environment, is:

```bash
python -m pip install --upgrade pip setuptools wheel packaging ninja psutil
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
BLOCK_SPARSE_ATTN_CUDA_ARCHS='80;90' MAX_JOBS=2 NVCC_THREADS=2 \
  BLOCK_SPARSE_ATTN_FORCE_BUILD=TRUE python -m pip install \
  --no-build-isolation --no-deps ./Block-Sparse-Attention
mkdir -p inputs results
```

Install PyTorch before the other requirements: `requirements.txt` pins CUDA-specific builds. `--no-build-isolation` lets the extension compile against the selected PyTorch. Building from the bundled source retains this project's backend changes.

## Updating an installation

From the checkout and active Python environment:

```bash
git pull --ff-only
bash scripts/install.sh
```

Editable installation exposes Python source changes immediately. Rebuild the attention backend after changes to its source, PyTorch, Python, or CUDA. Download model weights only when they change or are missing.

After updating from a version that bundled sample videos, run `python scripts/download_samples.py` to restore the default sample if needed. The installer downloads neither samples nor model weights.

## Troubleshooting

### CUDA or extension build errors

- Check `nvcc --version`, `CUDA_HOME`, and `python -c "import torch; print(torch.__version__, torch.version.cuda)"`.
- Ensure the selected compiler is CUDA 12.4+ within the 12.x series. CUDA 12.4 most closely matches the pinned PyTorch runtime.
- If compilation is killed because RAM is exhausted, rerun with `MAX_JOBS=1 NVCC_THREADS=1`.
- If the compiler rejects a GPU architecture, use `BLOCK_SPARSE_ATTN_CUDA_ARCHS='80;90'` with the baseline toolkit.
- If imports report undefined symbols, rerun the installer in the same Python environment used for inference to rebuild the extension.
- The backend imports PyTorch before its CUDA extension so PyTorch's libraries are loaded without a fixed environment-specific library path.

### Video tools

Both `ffmpeg` and `ffprobe` must be available in the process environment:

```bash
ffmpeg -version
ffprobe -version
```

To check whether the system encoder can actually use NVENC:

```bash
ffmpeg -hide_banner -f lavfi -i color=size=128x128:rate=1 \
  -frames:v 1 -c:v h264_nvenc -f null -
```

Inference checks encoder listings in both the system FFmpeg and imageio's FFmpeg. Listed encoders can still fail if driver libraries or hardware are unavailable. The system FFmpeg needs `libx264` for software encoding. To make imageio use that same executable, set `IMAGEIO_FFMPEG_EXE` to the absolute path returned by `command -v ffmpeg`.

### CUDA memory exhaustion

Use `--mode tiny --tile-dit`, reduce `--tile-size` to 128, or shorten the input. Full mode can also use `--tile-vae`. For long files, use `long_video_worker.py` with shorter segments. BF16 and FP16 both use two bytes per element, so switching between them alone does not halve memory usage.

### Missing weights or audio

Verify the filenames in the weights table and `FLASHVSR_MODEL_PATH`. If models were obtained through Git LFS, run `git lfs pull` in that model checkout. For audio, pass `--keep-audio` and check `ffprobe -i input.mp4` for an audio stream.

### Missing sample or failed download

Run `python scripts/download_samples.py --list` to see valid sample paths, then download the required one. If an existing file differs from the manifest, choose another `--output-dir` or use `--force` to replace it after the new download passes verification. Interrupted or invalid downloads are discarded; rerun the same command to retry. Using your own input does not require access to the sample archive.

When reporting a problem, include the Linux/Colab runtime, Python version, GPU model, `nvcc` version, PyTorch version, command, and full error output.
