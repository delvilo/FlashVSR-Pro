# FlashVSR-Pro installation

This guide covers direct installation and execution on **native Linux and Google Colab**. Windows, macOS, and WSL 2 are not supported.

## Requirements

- Linux with an NVIDIA driver and an Ampere, Ada, or Hopper GPU.
- Python 3.13–3.14. Python 3.13 is the Linux/Colab reference version.
- CUDA Toolkit **12.8 or newer (12.x)**, including `nvcc`, with PyTorch **2.11.0+cu128**. Toolkit 12.8 matches the PyTorch wheels; compiling with another CUDA 12.x minor version may emit a PyTorch minor-version warning. CUDA 13.x is incompatible with building extensions against cu128 wheels.
- A C++17 compiler, Git, FFmpeg, and FFprobe.
- Enough disk space for model weights, build files, input videos, and results.
- VRAM requirements vary with resolution and duration. Begin with a short clip and `--tile-dit`.

The bundled attention backend cannot run on Colab T4/P100 GPUs. The pinned setup targets Ampere/Ada/Hopper; newer GPU generations require a separate dependency upgrade.

## Linux

### 1. System packages

For distributions with Python 3.13 packages:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git git-lfs ffmpeg python3.13-venv python3.13-dev
```

Install the NVIDIA driver and CUDA Toolkit using [NVIDIA's Linux installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). CUDA 12.8 matches the wheel; later 12.x toolkits are allowed. Verify:

```bash
nvidia-smi
nvcc --version
```

The CUDA version displayed by `nvidia-smi` describes driver compatibility; `nvcc --version` identifies the installed compiler. A working driver alone is insufficient to build the extension.

If CUDA is outside `PATH`, set its actual location, for example:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
```

### 2. Checkout and Python environment

```bash
git clone --depth 1 https://github.com/delvilo/FlashVSR-Pro.git
cd FlashVSR-Pro
python3.13 --version
python3.13 -m venv .venv
source .venv/bin/activate
```

If Python 3.13 is unavailable in the system package repository, install it with matching development and venv packages or create a Python 3.13–3.14 Conda environment before installing. Python 3.12 and earlier are no longer supported.

The sparse attention source and CUTLASS headers are included as regular files in this checkout. No submodule initialization is required. See [THIRD_PARTY.md](THIRD_PARTY.md) for dependency sources and licenses. The shallow clone avoids downloading historical sample binaries and generated documentation; existing Git history is unchanged.

### 3. Install

```bash
bash scripts/install.sh
```

The installer uses the active `python` interpreter, checks prerequisites, installs PyTorch 2.11.0 from the CUDA 12.8 index, installs the application dependencies, and builds the bundled CUDA extension against that PyTorch installation. The minimal runtime does not require OpenCV, torchvision, torchaudio or Transformers. It creates `inputs/` and `results/`, checks dependency consistency and application imports, and records installed versions in `results/environment.json`. Use a dedicated environment because installation changes its packages.

To select an explicit interpreter:

```bash
FLASHVSR_PYTHON=/path/to/venv/bin/python bash scripts/install.sh
```

Build settings can be overridden:

```bash
MAX_JOBS=2 NVCC_THREADS=2 BLOCK_SPARSE_ATTN_CUDA_ARCHS='80;90' bash scripts/install.sh
```

The defaults limit compiler memory usage and select Ampere/Hopper architectures supported by the CUDA 12.8+ baseline. Blackwell `100`, `110`, or `120` targets require matching hardware and separate GPU acceptance; they are not covered by the default build.

### 4. Download weights

```bash
python -m flashvsr models list --mode all
python -m flashvsr models download --mode all
python -m flashvsr models check --mode all
```

Use `--mode tiny` or `--mode tiny-long` to skip the full-mode VAE, or `--mode full`
to skip TCDecoder. The default directory is `~/.cache/flashvsr/v1.1`.
`XDG_CACHE_HOME` changes the base cache location; `FLASHVSR_CACHE_DIR` changes
the FlashVSR cache root. `--model-dir` overrides `FLASHVSR_MODEL_PATH` and
selects an exact model directory. Every workflow uses the same rules.

| File in the selected directory | Used by |
| --- | --- |
| `diffusion_pytorch_model_streaming_dmd.safetensors` | All modes |
| `LQ_proj_in.ckpt` | All modes |
| `TCDecoder.ckpt` | `tiny`, `tiny-long` |
| `Wan2.1_VAE.pth` | `full` |
| `posi_prompt.pth` | All modes; copied from the installed package |

The versioned manifest pins an upstream commit, byte size and SHA-256 for each
file. Downloads verify temporary files before atomic replacement. Re-running
the command skips valid files and repairs corrupt ones. Interrupted downloads
remove partial data; a retry restarts the affected file. Inference verifies
models before importing PyTorch and never downloads them automatically.

Existing installations can reuse their weights without moving them:

```bash
export FLASHVSR_MODEL_PATH="$PWD/models/FlashVSR-v1.1"
python -m flashvsr models download --mode all
python -m flashvsr models check --mode all
```

This checks existing weights and adds the prompt to the same directory. The
legacy `FLASHVSR-Pro_MODEL_PATH` environment key is still accepted. See
[ARCHITECTURE.md](ARCHITECTURE.md) for integrity and cache behavior.

### 5. Verify and run

```bash
python -c "import torch; import block_sparse_attn; import diffsynth; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
python -m flashvsr infer --help
python scripts/download_samples.py example0.mp4
python -m flashvsr infer -i inputs/example0.mp4 -o results/ --mode tiny --scale 4.0 --tile-dit
```

The sample downloader needs only Python's standard library. It fetches the selected video from a fixed repository commit and verifies its byte count and SHA-256 before saving it. Use your own video path to skip the sample download. See [inputs/README.md](inputs/README.md) for the optional collection.

For subsequent sessions, activate the same environment. You may call the scripts using absolute paths from another directory; the default model cache does not depend on the shell's working directory.

## Google Colab

Use [colab/FlashVSR_Pro.ipynb](colab/FlashVSR_Pro.ipynb) with **A100 or L4**, **Ubuntu 24.04 noble**, native **Python 3.13**, **PyTorch 2.11.0+cu128** and **CUDA Toolkit 12.8**. This notebook does not support older Colab images or T4/P100. A driver reporting CUDA 13.0 in `nvidia-smi` is compatible with this baseline; the toolkit is checked using `nvcc`.

The notebook mounts Drive, diagnoses the environment, preserves Colab's matching PyTorch, and runs each stage in a fresh subprocess. It uses the native interpreter and headers, without a virtual environment or Ubuntu's potentially mismatched `python3-dev` package. Use the notebook setup command for persistent caching instead of the Linux installer.

Models and compiled attention wheels are verified and cached in `MyDrive/FlashVSR-Pro`. Video processing uses local `/content/flashvsr` storage; every completed segment and the final result are verified and saved to Drive. The default project budget is **20 GB including a 2 GB reserve**. Lossless segments can exhaust this budget; the job stops and preserves checkpoints for resume.

From the checkout after mounting Drive:

```bash
python -m flashvsr.colab diagnose
python -m flashvsr.colab setup --mode tiny
python -m flashvsr.colab status
python -m flashvsr.colab run --job example --input /content/drive/MyDrive/FlashVSR-Pro/inputs/input.mp4
# Restore the same code and settings before resuming:
python -m flashvsr.colab run --job example --resume
```

See [COLAB.md](COLAB.md) for notebook settings, cache compatibility, Drive paths, budget management, runtime-reset recovery, result retention and optional A100/L4 GPU acceptance.

## Manual dependency installation

The installer is preferred because it validates the selected environment. The equivalent installation order, from the checkout in a supported Python environment, is:

```bash
python -m pip install -r requirements-build.txt
python -m pip install -r requirements-cuda.txt
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --no-deps -e .
BLOCK_SPARSE_ATTN_CUDA_ARCHS='80;90' MAX_JOBS=2 NVCC_THREADS=2 \
  BLOCK_SPARSE_ATTN_FORCE_BUILD=TRUE python -m pip install \
  --no-build-isolation --no-deps ./Block-Sparse-Attention
mkdir -p inputs results
python -m pip check
```

Install PyTorch before the other requirements: `requirements-cuda.txt` selects the official cu128 index and exact CUDA wheel versions. `requirements.txt` supplies runtime version pins without an index URL, so package metadata remains valid and CPU wheels can be used for unit tests. `--no-build-isolation` lets the extension compile against the selected PyTorch. Building from the bundled source retains this project's backend changes.

## Package and development layout

`pyproject.toml` holds the package metadata, pinned build backend, and application dependencies from `requirements.txt`; `setup.py` is a compatibility shim and enforces the Python/Linux baseline. Dependency groups are kept in separate files:

| File | Purpose |
| --- | --- |
| `requirements.txt` | Eight pinned direct runtime dependencies |
| `requirements-cuda.txt` | PyTorch cu128 wheel for inference |
| `requirements-build.txt` | Pinned installer and CUDA build tools |
| `requirements-dev.txt` | CPU test and package-build tools |
| `requirements-test-torch.txt` | CPU PyTorch wheels for tests only |

The reference combination is Python **3.13.x**, PyTorch **2.11.0+cu128**, and CUDA Toolkit **12.8.x**. Native Linux also accepts Python 3.14 and later 12.x toolkits; the Colab workflow requires the exact reference baseline. PyTorch's extension builder warns when `nvcc` and its wheels use different CUDA 12.x minor versions. CPU CI covers Python 3.13–3.14. Other direct Python dependencies and build tools use recent compatible pins; upgrading PyTorch further needs real GPU compilation and inference validation of the bundled extension. Direct pins do not lock the Linux driver, system libraries, or every transitive dependency. Keep the generated environment report with GPU acceptance results when reproducing a run.

The CUDA wheel versions are available from the [official PyTorch CUDA 12.8 wheel index](https://download.pytorch.org/whl/cu128/torch/). Unused generation, training, tokenizer, OpenCV and audio-framework packages have been removed from the application requirements.

The wheel provides the `flashvsr` console command, library modules, model manifest and fixed prompt; installed commands work outside the checkout. Large model weights and the compiled attention backend remain separate. The source distribution additionally includes legacy scripts, tests, the Colab notebook and CUDA build sources. The installation script uses an editable checkout for development. See [TESTING.md](TESTING.md) for building and validating both distributions.

## Updating an installation

For native Linux, from the checkout and active Python environment:

```bash
git pull --ff-only
bash scripts/install.sh
```

Editable installation exposes Python source changes immediately. Rebuild the attention backend after changes to its source, PyTorch, Python, or CUDA. Run `python -m flashvsr models check` after updates; `models download` repairs missing or corrupt weights.

For Colab, select the revision in the notebook and rerun setup. Cache keys select compatible build artifacts automatically. Keep the saved code/runtime/settings when resuming an unfinished job; use a new job name after an upgrade.

After updating from a version that bundled sample videos, run `python scripts/download_samples.py` to restore the default sample if needed. The installer downloads neither samples nor model weights.

## Troubleshooting

### CUDA or extension build errors

- Check `nvcc --version`, `CUDA_HOME`, and `python -c "import torch; print(torch.__version__, torch.version.cuda)"`.
- Ensure the selected compiler is CUDA 12.8 or newer within the 12.x series. A 12.x minor mismatch with the cu128 wheel produces a PyTorch warning; CUDA 13.x is unsupported by this baseline.
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

To check whether the system encoder can use the requested HEVC NVENC settings:

```bash
ffmpeg -hide_banner -f lavfi -i color=size=128x128:rate=30:duration=2 \
  -map 0:v:0 -c:v hevc_nvenc -preset p7 -tune hq -rc vbr -cq 20 -b:v 0 \
  -multipass fullres -bf 3 -b_ref_mode middle -rc-lookahead 32 \
  -spatial-aq 1 -temporal-aq 1 -pix_fmt yuv420p -frames:v 40 -f null -
```

The application detects the selected FFmpeg version and uses `-fps_mode cfr` on version 5.1 and later, or legacy `-vsync 1` on older releases. For MP4/MOV/MKV it tests a full 40-frame HEVC NVENC encode with all selected settings before processing the video. Default quality 10 uses `-cq 20`; the other quality settings map to CQ 20–26. If the selected FFmpeg, driver, or GPU cannot use the settings, or if actual encoding or output verification fails, the output uses `libx264` instead. AVI retains H.264 NVENC; WebM uses `libvpx-vp9`; GIF uses GIF, and odd dimensions use software encoding. Every output must pass codec, frame count, FPS, dimension, and requested audio-track checks before replacing an existing file. All audio tracks are preserved when requested (AAC, or Opus for WebM); subtitles and attachments are not copied. Set `FLASHVSR_FFMPEG` and `FLASHVSR_FFPROBE` to explicit executable paths to override `PATH`. Imageio decoding is configured to use that same selected FFmpeg, so `IMAGEIO_FFMPEG_EXE` is not a separate encoder choice.

### CUDA memory exhaustion

Use `--mode tiny --tile-dit`, reduce `--tile-size` to 128, or shorten the input. Full mode can also use `--tile-vae`. For long files, use `flashvsr long --segment-frames 65 --work-dir /data/jobs/clip` with the usual input/output arguments. A failed job keeps completed segments; repeat the same command with `--resume`. Changed settings need a new job directory. BF16 and FP16 both use two bytes per element, so switching between them alone does not halve memory usage.

### Missing weights or audio

Verify the filenames in the weights table and `FLASHVSR_MODEL_PATH`. If models were obtained through Git LFS, run `git lfs pull` in that model checkout. For audio, pass `--keep-audio` and check `ffprobe -i input.mp4` for an audio stream.

### Missing sample or failed download

Run `python scripts/download_samples.py --list` to see valid sample paths, then download the required one. If an existing file differs from the manifest, choose another `--output-dir` or use `--force` to replace it after the new download passes verification. Interrupted or invalid downloads are discarded; rerun the same command to retry. Using your own input does not require access to the sample archive.

When reporting a problem, include the Linux/Colab runtime, Python version, GPU model, `nvcc` version, PyTorch version, command, and full error output.

### Logs and reports

Every inference writes `OUTPUT.json`; batches also write `batch.json`, and long
videos write a merged-output report with per-segment details. Use `--log-file results/run.jsonl --log-level DEBUG` for structured diagnostic logs. Reports
include model identity, arguments, timings, FPS and peak CUDA memory. Retain
these files with the generated `results/environment.json` when reporting an issue.
