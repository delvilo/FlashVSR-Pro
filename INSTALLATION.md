# FlashVSR-Pro installation

This guide covers direct installation and execution on **native Linux and Google Colab**. Windows, macOS, and WSL 2 are not supported.

## Requirements

- Linux with an NVIDIA driver and an Ampere, Ada, or Hopper GPU.
- Python 3.10–3.12. Python 3.11 is the Linux baseline.
- CUDA Toolkit **12.4.x**, including `nvcc`, matching PyTorch **2.6.0+cu124**, torchvision **0.21.0+cu124**, and torchaudio **2.6.0+cu124**. A different toolkit minor version is rejected before package installation.
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

Install the NVIDIA driver and CUDA Toolkit using [NVIDIA's Linux installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). Select CUDA 12.4 to match the pinned PyTorch build. Verify:

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

The installer uses the active `python` interpreter, checks prerequisites, installs PyTorch 2.6.0/torchvision 0.21.0/torchaudio 2.6.0 from the CUDA 12.4 index, installs the application dependencies, and builds the bundled CUDA extension against that PyTorch installation. It removes conflicting OpenCV distributions and installs only `opencv-python-headless`. It creates `inputs/` and `results/`, checks dependency consistency and application imports, and records installed versions in `results/environment.json`. Use a dedicated environment because installation changes its packages.

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

The notebook creates `project/.venv` using its Python version, then uses that environment's interpreter for installation, model downloads, inference, and GPU acceptance. This isolates the pinned dependencies from Colab's preinstalled packages. No Conda activation or notebook-kernel restart is needed.

The runtime needs Python 3.10–3.12 and a CUDA 12.4.x toolkit. If a runtime's Python or toolkit is outside that range, use a compatible runtime or install the baseline environment before proceeding. Check toolkit availability with `nvcc --version`; do not substitute the driver version from `nvidia-smi`.

### Manual cells

First clone the code and install system packages:

```python
from pathlib import Path
import subprocess
import sys

project = Path('/content/FlashVSR-Pro')
subprocess.run(['apt-get', 'update'], check=True)
subprocess.run(['apt-get', 'install', '-y', 'build-essential', 'git', 'git-lfs', 'ffmpeg', 'python3-dev', 'python3-venv'], check=True)
if not project.exists():
    subprocess.run(['git', 'clone', '--depth', '1',
                    'https://github.com/delvilo/FlashVSR-Pro.git', str(project)], check=True)
```

Create and install into a dedicated Python environment:

```python
import os

venv = project / '.venv'
if not (venv / 'bin/python').exists():
    subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
project_python = str(venv / 'bin/python')
install_env = os.environ.copy()
install_env['FLASHVSR_PYTHON'] = project_python
# If needed, select an installed compatible toolkit:
# install_env['CUDA_HOME'] = '/usr/local/cuda-12.4'
subprocess.run(['bash', 'scripts/install.sh'], cwd=project, env=install_env, check=True)
```

Download models and choose an input. The following cell uses an optional sample; replace `input_video` with an uploaded video or a path on mounted Google Drive to use your own:

```python
subprocess.run([
    project_python, '-c',
    "from huggingface_hub import snapshot_download; "
    "snapshot_download('JunhaoZhuang/FlashVSR-v1.1', local_dir='models/FlashVSR-v1.1')"
], cwd=project, check=True)

input_video = project / 'inputs/example0.mp4'  # Or Path('/content/input.mp4').
output_video = Path('/content/enhanced.mp4')
if input_video == project / 'inputs/example0.mp4':
    subprocess.run([
        project_python, str(project / 'scripts/download_samples.py'), 'example0.mp4'
    ], cwd=project, check=True)
if not input_video.is_file():
    raise FileNotFoundError(input_video)
subprocess.run([
    project_python, str(project / 'infer.py'),
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
python -m pip install -r requirements-build.txt
python -m pip install -r requirements-cuda.txt
python -m pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless opencv-python-headless
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -e .
BLOCK_SPARSE_ATTN_CUDA_ARCHS='80;90' MAX_JOBS=2 NVCC_THREADS=2 \
  BLOCK_SPARSE_ATTN_FORCE_BUILD=TRUE python -m pip install \
  --no-build-isolation --no-deps ./Block-Sparse-Attention
mkdir -p inputs results
python -m pip check
```

Install PyTorch before the other requirements: `requirements-cuda.txt` selects the official cu124 index and exact CUDA wheel versions. `requirements.txt` supplies runtime version pins without an index URL, so package metadata remains valid and CPU wheels can be used for unit tests. `--no-build-isolation` lets the extension compile against the selected PyTorch. Building from the bundled source retains this project's backend changes.

## Package and development layout

`pyproject.toml` holds the package metadata and pinned build backend; `setup.py` is a Linux compatibility shim. Dependency groups are kept in separate files:

| File | Purpose |
| --- | --- |
| `requirements.txt` | Pinned direct application dependencies; one headless OpenCV build |
| `requirements-cuda.txt` | PyTorch/torchvision/torchaudio cu124 wheels for inference |
| `requirements-build.txt` | Pinned installer and CUDA build tools |
| `requirements-dev.txt` | CPU test and package-build tools |
| `requirements-test-torch.txt` | CPU PyTorch wheels for tests only |

The reference combination is Python **3.11.x**, PyTorch **2.6.0+cu124**, and CUDA Toolkit **12.4.x**. Python 3.10.x and 3.12.x are also covered by CPU CI. Direct dependency and build-tool versions are pinned; this does not lock the Linux driver, system libraries, or every transitive Python dependency. Keep the generated environment report with GPU acceptance results when reproducing a run.

The CUDA wheel versions follow the [official PyTorch 2.6 installation matrix](https://pytorch.org/get-started/previous-versions/). OpenCV's distributions share the `cv2` namespace, so the installer removes the variants and installs only the pinned headless package.

Run the command-line programs from a checkout or unpacked source distribution with an editable install. The source distribution includes scripts, the prompt tensor, and CUDA build sources. The Python wheel supplies the library modules and tokenizer data; it is not a standalone bundle of models, command-line scripts, and the compiled attention backend. See [TESTING.md](TESTING.md) for building and validating both distributions.

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
- Ensure the selected compiler is CUDA 12.4.x. A newer driver is acceptable, but the compiler used to build the extension must match the cu124 baseline.
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

Inference performs a real NVENC encode with the selected FFmpeg and retries failed hardware encodes with `libx264`. WebM uses `libvpx-vp9`; odd dimensions use software encoding. The shared media code verifies the result before replacing an existing output. All audio tracks are preserved when requested (AAC, or Opus for WebM); subtitles and attachments are not copied. Set `FLASHVSR_FFMPEG` and `FLASHVSR_FFPROBE` to explicit executable paths to override `PATH`. Imageio decoding is configured to use that same selected FFmpeg, so `IMAGEIO_FFMPEG_EXE` is not a separate encoder choice.

### CUDA memory exhaustion

Use `--mode tiny --tile-dit`, reduce `--tile-size` to 128, or shorten the input. Full mode can also use `--tile-vae`. For long files, use `long_video_worker.py` with shorter segments. BF16 and FP16 both use two bytes per element, so switching between them alone does not halve memory usage.

### Missing weights or audio

Verify the filenames in the weights table and `FLASHVSR_MODEL_PATH`. If models were obtained through Git LFS, run `git lfs pull` in that model checkout. For audio, pass `--keep-audio` and check `ffprobe -i input.mp4` for an audio stream.

### Missing sample or failed download

Run `python scripts/download_samples.py --list` to see valid sample paths, then download the required one. If an existing file differs from the manifest, choose another `--output-dir` or use `--force` to replace it after the new download passes verification. Interrupted or invalid downloads are discarded; rerun the same command to retry. Using your own input does not require access to the sample archive.

When reporting a problem, include the Linux/Colab runtime, Python version, GPU model, `nvcc` version, PyTorch version, command, and full error output.
