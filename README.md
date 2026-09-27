# FlashVSR-Pro: Real-Time Video Super-Resolution

FlashVSR-Pro is an independent implementation of the diffusion-based streaming video super-resolution method from [FlashVSR](https://arxiv.org/abs/2510.12747). It provides a unified command-line interface for enhancing videos and image sequences.

FlashVSR-Pro supports direct Python execution on **native Linux and Google Colab**. See [INSTALLATION.md](INSTALLATION.md) for setup, model downloads, and troubleshooting.

## Features

- **Unified inference:** `infer.py` selects `full`, `tiny`, or `tiny-long` mode.
- **Audio preservation:** `--keep-audio` transfers all input audio tracks to the processed video.
- **Lower VRAM use:** `--tile-dit` splits inference into overlapping tiles; `--tile-vae` enables tiled decoding in full mode.
- **Video alignment:** spatial and temporal padding accommodate model constraints, then output is cropped to the requested resolution and adjusted to the input frame count.
- **GPU acceleration:** CUDA inference, TF32/cuDNN optimizations, and NVENC encoding when available, with software encoding support.
- **Native installation:** `scripts/install.sh` installs the pinned Python dependencies and builds the bundled Block-Sparse-Attention backend.
- **Additional workflows:** `batch_inference.py` processes a video directory; `long_video_worker.py` splits, processes, and rejoins long videos.

## Supported setup

| Component | Baseline |
| --- | --- |
| System | Linux; Google Colab with a compatible GPU runtime |
| Python | 3.12.x reference; 3.13.x and 3.14.x compatibility targets |
| PyTorch | 2.10.0 with CUDA 12.6 wheels |
| CUDA Toolkit | 12.5 or newer (12.x); 12.6 matches the cu126 wheels |
| GPU | NVIDIA Ampere, Ada, or Hopper: for example RTX 3090/4090, A100, H100, or L4 |
| Build tools | C++17 compiler, Ninja, packaging, and wheel |
| Video tools | FFmpeg and FFprobe on `PATH` |

VRAM use depends on resolution, duration, mode, and tiling. Start with a short clip in `tiny` mode and `--tile-dit`; 16 GB or more gives more room for processing. A GPU being assigned in Colab does not guarantee compatibility: **T4 and P100 are unsupported** by this backend. Select A100 or L4 when available.

Windows, macOS, and WSL 2 are not supported. Project build scripts target Linux; bundled third-party source retains its upstream portability code.

## Linux quick start

Install an NVIDIA driver and the CUDA Toolkit using [NVIDIA's Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). The toolkit supplies `nvcc`; PyTorch wheels alone do not supply the compiler.

On a Linux distribution with Python 3.12 development and venv packages (for example Ubuntu 24.04), install system dependencies and create a Python environment:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git git-lfs ffmpeg python3.12-venv python3.12-dev

git clone --depth 1 https://github.com/delvilo/FlashVSR-Pro.git
cd FlashVSR-Pro

# Use Python 3.12 (and its matching venv/dev packages) if the system Python differs.
python3.12 -m venv .venv
source .venv/bin/activate
bash scripts/install.sh
```

An existing Conda environment with a supported Python version can also be used. Activate it before invoking the installer; all subprocesses use the selected Python environment.
On distributions without Python 3.12 packages, install a matching interpreter and headers first or use a Python 3.12–3.14 Conda environment.

Download the model weights into the project checkout:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('JunhaoZhuang/FlashVSR-v1.1', local_dir='models/FlashVSR-v1.1')"
```

Download and run a short sample:

```bash
python scripts/download_samples.py example0.mp4
python infer.py -i inputs/example0.mp4 -o results/ --mode tiny --scale 4.0 --tile-dit
```

Sample videos are optional downloads. Use your own input directly, or see [inputs/README.md](inputs/README.md) to list and download other samples. Downloads are checked against a pinned size and SHA-256 manifest.

After opening a new shell, activate the same environment before running inference. Installation does not need to be repeated unless dependencies change.

## Google Colab

Open [colab/FlashVSR_Pro.ipynb](colab/FlashVSR_Pro.ipynb) in Google Colab, select an **A100 or L4 GPU runtime**, and run the cells in order. The notebook checks the GPU and toolkit, creates a dedicated Python environment in the checkout, downloads models, and runs `infer.py` as a subprocess. It downloads the default sample on demand; you can also select an uploaded video. Use a short input first.

The complete manual workflow is in [INSTALLATION.md — Google Colab](INSTALLATION.md#google-colab). Colab's runtime storage is temporary; download results or copy them to mounted Google Drive before the runtime is reset.

## Inference modes

| Mode | Decoder | Purpose |
| --- | --- | --- |
| `full` | Wan2.1 VAE | Quality-oriented processing with higher VRAM use |
| `tiny` | TCDecoder | Balanced speed, quality, and VRAM use; default |
| `tiny-long` | TCDecoder | Streaming model path for longer videos |

```bash
python scripts/download_samples.py example0.mp4 example4.mp4
python infer.py -i inputs/example0.mp4 -o results/ --mode full --tile-vae
python infer.py -i inputs/example0.mp4 -o results/ --mode tiny --keep-audio
python infer.py -i inputs/example4.mp4 -o results/ --mode tiny-long --tile-dit
```

Long videos can still consume substantial memory during input preparation. Use `long_video_worker.py` to split large inputs into smaller segments when necessary.

## Arguments

| Argument | Purpose | Default |
| --- | --- | --- |
| `-i`, `--input` | Video file or directory of PNG/JPEG frames | Required |
| `-o`, `--output` | Output directory or video file | `./results` |
| `--mode` | `full`, `tiny`, or `tiny-long` | `tiny` |
| `--scale` | Resolution multiplier | `2.0` |
| `--keep-audio` | Preserve the input audio | Off |
| `--tile-dit` | Enable tiled DiT inference | Off |
| `--tile-vae` | Enable tiled VAE decoding in full mode | Off |
| `--tile-size` | Tile size | `256` |
| `--overlap` | Overlap between tiles | `24` |
| `--color-fix` | Apply color correction | Off |
| `--fps` | Output frame rate | Input FPS, or 30 for images |
| `--quality` | Output video quality, 0–10 | `10` |
| `--dtype` | `bf16` or `fp16` | `bf16` |
| `--device` | Processing device; CUDA is required by the sparse backend | `cuda` |
| `--seed` | Random seed | `0` |
| `--sparse-ratio` | Sparse attention ratio | `2.0` |
| `--kv-ratio` | KV cache ratio | `3.0` |
| `--local-range` | Local attention range | `11` |
| `--metrics-json` | Write GPU memory, timing and verified output metadata | Off |

The model was designed for 4× upscaling; use `--scale 4.0` for that workflow. Run `python infer.py --help` for the command-line reference. Sparse attention requires FP16/BF16 and a compatible CUDA GPU. Unsupported devices, precision, parameter values, and missing model files stop with an error; there is no CPU inference fallback.

An existing directory, or an output path without a suffix, is treated as an output directory. File paths such as `results/enhanced.mp4` select an explicit output filename. User-supplied input/output paths are relative to the current working directory. Default model and prompt paths are relative to the project files.

To store model weights elsewhere:

```bash
export FLASHVSR_MODEL_PATH=/data/models/FlashVSR-v1.1
python infer.py -i /data/input.mp4 -o /data/results/ --mode tiny --keep-audio
```

The model path applies to the DiT, LQ projector, and decoder weights. The fixed prompt stays in `models/prompt_tensor/posi_prompt.pth` in the checkout. The legacy `FLASHVSR-Pro_MODEL_PATH` environment key remains readable for existing callers.

## Batch and long-video programs

### Batch directory

```bash
python batch_inference.py
```

Place your videos in `inputs/`, or download selected samples first. This recursively reads the checkout's `inputs/` directory and writes corresponding subdirectories under `results/`. It defaults to tiny mode and 2× scaling. Use `--input-dir`, `--output-dir`, `--mode`, and `--scale` to change defaults. The output directory must be outside the input tree. Edit `EXCLUDE_FILES` and `SPECIAL_CONFIGS` in the script for per-file choices. A failed child makes the batch exit with status 1 after processing the remaining files. Child processes use the same Python interpreter as the batch program.

### Split and rejoin a long video

```bash
python long_video_worker.py -i /data/long.mp4 -o /data/results \
  --segment_time 00:01:00 --mode tiny --scale 2.0
```

The worker uses FFmpeg to split the source, calls `infer.py` for each segment with audio preservation, and merges the results. Splits occur at existing keyframes, so the requested duration is approximate. Each run uses a new temporary directory that is removed on success or failure. Pass `--keep-temp` to preserve segments for diagnosis. Each segment and the merged output are verified before the final file replaces an existing result.

## Project files

| File or directory | Role |
| --- | --- |
| `infer.py` | Main inference entry point |
| `batch_inference.py` | Recursive batch processing |
| `long_video_worker.py` | Segment processing and concatenation |
| `scripts/install.sh` | Linux/Colab dependency and CUDA extension installation |
| `scripts/download_samples.py` | Optional sample downloads with size and SHA-256 verification |
| `inputs/samples.json`, `inputs/README.md` | Pinned sample manifest and download instructions |
| `colab/FlashVSR_Pro.ipynb` | Colab setup and inference notebook |
| `diffsynth/pipelines/flashvsr_*.py` | Three inference pipelines |
| `utils/` | Decoders, audio, tiling, and model loading |
| `Block-Sparse-Attention/` | Bundled sparse attention source and CUTLASS headers |
| `models/` | Downloaded weights and bundled prompt tensor |
| `pyproject.toml`, `setup.py` | Package metadata and compatibility entry point |
| `requirements*.txt` | Separate runtime, CUDA, build, and CPU development dependencies |
| `scripts/validate_gpu.py`, `TESTING.md` | Three-mode GPU acceptance and test instructions |
| `INSTALLATION.md` | Setup, model files, updates, and troubleshooting |
| `THIRD_PARTY.md` | Bundled dependency sources, licenses, and maintenance notes |

Generated CUTLASS HTML documentation and sample video binaries are excluded from the current source tree. See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream documentation and the retained build sources. Vendored CUTLASS files are marked as third-party code for GitHub language statistics.

Use the shallow clone command above for a smaller initial download. Older commits still contain the removed files; this cleanup does not rewrite Git history or reduce an existing checkout's `.git` directory.

## Troubleshooting

- **Missing CUDA compiler or library:** check `nvcc --version`, `CUDA_HOME`, and the NVIDIA driver. Rebuild the backend after changing PyTorch or CUDA.
- **Colab receives a T4/P100:** select a supported runtime. Installing another CUDA package cannot add missing GPU capabilities.
- **Out of memory:** enable `--tile-dit`, reduce tile size (minimum 128), use `tiny`, or shorten segments. Use `--tile-vae` with full mode.
- **Missing model files:** download the model repository into `models/FlashVSR-v1.1`, or set `FLASHVSR_MODEL_PATH`.
- **Missing audio:** pass `--keep-audio`, check that the input has audio, and ensure both `ffmpeg` and `ffprobe` are installed.
- **NVENC issues:** the selected FFmpeg must complete a real test encode before NVENC is used. If hardware encoding fails, output is retried with libx264. See [INSTALLATION.md](INSTALLATION.md#video-tools).

## Reliability checks

The three command-line programs use exit codes **0** (success), **1** (runtime, input, model, or encoding failure), **2** (invalid arguments), and **130** (interrupted). `--help` works before installing the inference dependencies. Model loaders reject missing, empty, Git LFS pointer, corrupt, and incompatible weights instead of proceeding with uninitialized parameters.

One FFmpeg implementation handles encoding, audio muxing, and concatenation. Outputs are verified for dimensions, frame count, and audio tracks before an atomic replacement; failed encodes preserve an existing output. `--keep-audio` preserves every audio track, re-encoding audio to AAC (Opus for WebM). Inputs without audio produce a silent output. Subtitles and attachments are not copied. GIF output cannot preserve audio.

See [TESTING.md](TESTING.md) for CPU CI, source-package checks, and real GPU acceptance. CPU tests do not certify CUDA kernel or model execution.

## License and attribution

This project is distributed under the [Apache 2.0 license](LICENSE). Bundled dependencies retain their own licenses.

If using the algorithm in research, cite:

```bibtex
@article{zhuang2025flashvsr,
  title={FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution},
  author={Zhuang, Junhao and Guo, Shi and Cai, Xin and Li, Xiaohui and Liu, Yihao and Yuan, Chun and Xue, Tianfan},
  journal={arXiv preprint arXiv:2510.12747},
  year={2025}
}
```

Credits: [original FlashVSR](https://github.com/OpenImagingLab/FlashVSR), [upstream FlashVSR-Pro](https://github.com/LujiaJin/FlashVSR-Pro), [FlashVSR_plus](https://github.com/lihaoyun6/FlashVSR_plus) for audio/tiling ideas, and [Block-Sparse-Attention](https://github.com/LujiaJin/Block-Sparse-Attention) for the optimized backend.

Contributions and issues for this fork: [delvilo/FlashVSR-Pro](https://github.com/delvilo/FlashVSR-Pro/issues).
