# FlashVSR-Pro: Real-Time Video Super-Resolution

FlashVSR-Pro is an independent implementation of the diffusion-based streaming video super-resolution method from [FlashVSR](https://arxiv.org/abs/2510.12747). It provides a unified command-line interface for enhancing videos and image sequences.

This branch uses direct Python execution on **Linux and Google Colab**. See [INSTALLATION.md](INSTALLATION.md) for setup, model downloads, and troubleshooting.

## Features

- **Unified inference:** `infer.py` selects `full`, `tiny`, or `tiny-long` mode.
- **Audio preservation:** `--keep-audio` transfers the input audio to the processed video.
- **Lower VRAM use:** `--tile-dit` splits inference into overlapping tiles; `--tile-vae` enables tiled decoding in full mode.
- **Video alignment:** spatial and temporal padding accommodate model constraints, then output is cropped to the requested resolution and adjusted to the input frame count.
- **GPU acceleration:** CUDA inference, TF32/cuDNN optimizations, and NVENC encoding when available, with software encoding support.
- **Native installation:** `scripts/install.sh` installs the pinned Python dependencies and builds the bundled Block-Sparse-Attention backend.
- **Additional workflows:** `batch_inference.py` processes a video directory; `long_video_worker.py` splits, processes, and rejoins long videos.

## Supported setup

| Component | Baseline |
| --- | --- |
| System | Linux; Google Colab with a compatible GPU runtime |
| Python | 3.10–3.12; Python 3.11 recommended for Linux |
| PyTorch | 2.6.0 with CUDA 12.4 wheels |
| CUDA Toolkit | 12.4 recommended; installer accepts 12.4+ within the 12.x series |
| GPU | NVIDIA Ampere, Ada, or Hopper: for example RTX 3090/4090, A100, H100, or L4 |
| Build tools | C++17 compiler, Ninja, packaging, and wheel |
| Video tools | FFmpeg and FFprobe on `PATH` |

VRAM use depends on resolution, duration, mode, and tiling. Start with a short clip in `tiny` mode and `--tile-dit`; 16 GB or more gives more room for processing. A GPU being assigned in Colab does not guarantee compatibility: **T4 and P100 are unsupported** by this backend. Select A100 or L4 when available.

WSL 2, native Windows, and macOS are outside this change's validation scope. Existing platform-specific code is deferred to a separate cleanup.

## Linux quick start

Install an NVIDIA driver and the CUDA Toolkit using [NVIDIA's Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). The toolkit supplies `nvcc`; PyTorch wheels alone do not supply the compiler.

On Ubuntu/Debian, install system dependencies and create a Python environment:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git git-lfs ffmpeg python3-venv python3-dev

git clone https://github.com/delvilo/FlashVSR-Pro.git
cd FlashVSR-Pro

# Use python3.11 (and its matching venv/dev packages) if python3 is outside 3.10–3.12.
python3 -m venv .venv
source .venv/bin/activate
bash scripts/install.sh
```

An existing Conda environment with a supported Python version can also be used. Activate it before invoking the installer; all subprocesses use the selected Python environment.

Download the model weights into the project checkout:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('JunhaoZhuang/FlashVSR-v1.1', local_dir='models/FlashVSR-v1.1')"
```

Run a short sample:

```bash
python infer.py -i inputs/example0.mp4 -o results/ --mode tiny --scale 4.0 --tile-dit
```

After opening a new shell, activate the same environment before running inference. Installation does not need to be repeated unless dependencies change.

## Google Colab

Open [colab/FlashVSR_Pro.ipynb](colab/FlashVSR_Pro.ipynb) in Google Colab, select an **A100 or L4 GPU runtime**, and run the cells in order. The notebook checks the GPU and toolkit, installs into the notebook's Python environment, downloads models, and runs `infer.py` as a subprocess. Use a short input first.

The complete manual workflow is in [INSTALLATION.md — Google Colab](INSTALLATION.md#google-colab). Colab's runtime storage is temporary; download results or copy them to mounted Google Drive before the runtime is reset.

## Inference modes

| Mode | Decoder | Purpose |
| --- | --- | --- |
| `full` | Wan2.1 VAE | Quality-oriented processing with higher VRAM use |
| `tiny` | TCDecoder | Balanced speed, quality, and VRAM use; default |
| `tiny-long` | TCDecoder | Streaming model path for longer videos |

```bash
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
| `--dtype` | `bf16`, `fp16`, or `fp32` | `bf16` |
| `--device` | Processing device; CUDA is required by the sparse backend | `cuda` |
| `--seed` | Random seed | `0` |
| `--sparse-ratio` | Sparse attention ratio | `2.0` |
| `--kv-ratio` | KV cache ratio | `3.0` |
| `--local-range` | Local attention range | `11` |

The model was designed for 4× upscaling; use `--scale 4.0` for that workflow. Run `python infer.py --help` for the command-line reference. Sparse attention supports FP16/BF16; the exposed FP32 option does not imply that every kernel supports FP32.

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

This recursively reads the checkout's `inputs/` directory and writes corresponding subdirectories under `results/`. It defaults to tiny mode and 2× scaling. Edit `EXCLUDE_FILES` and `SPECIAL_CONFIGS` in the script for per-file choices. Child processes use the same Python interpreter as the batch program.

### Split and rejoin a long video

```bash
python long_video_worker.py -i /data/long.mp4 -o /data/results \
  --segment_time 00:01:00 --mode tiny --scale 2.0
```

The worker uses FFmpeg to split the source, calls `infer.py` for each segment with audio preservation, and merges the results. Splits occur at existing keyframes, so the requested duration is approximate. It keeps temporary segments in `temp_work_<video-name>/` in the current directory. Remove them after checking the final output; rerunning the same video name replaces that working directory.

## Project files

| File or directory | Role |
| --- | --- |
| `infer.py` | Main inference entry point |
| `batch_inference.py` | Recursive batch processing |
| `long_video_worker.py` | Segment processing and concatenation |
| `scripts/install.sh` | Linux/Colab dependency and CUDA extension installation |
| `colab/FlashVSR_Pro.ipynb` | Colab setup and inference notebook |
| `diffsynth/pipelines/flashvsr_*.py` | Three inference pipelines |
| `utils/` | Decoders, audio, tiling, and model loading |
| `Block-Sparse-Attention/` | Bundled sparse attention source and CUTLASS headers |
| `models/` | Downloaded weights and bundled prompt tensor |
| `requirements.txt`, `setup.py` | Python dependencies and package metadata |
| `INSTALLATION.md` | Setup, model files, updates, and troubleshooting |

## Troubleshooting

- **Missing CUDA compiler or library:** check `nvcc --version`, `CUDA_HOME`, and the NVIDIA driver. Rebuild the backend after changing PyTorch or CUDA.
- **Colab receives a T4/P100:** select a supported runtime. Installing another CUDA package cannot add missing GPU capabilities.
- **Out of memory:** enable `--tile-dit`, reduce tile size (minimum 128), use `tiny`, or shorten segments. Use `--tile-vae` with full mode.
- **Missing model files:** download the model repository into `models/FlashVSR-v1.1`, or set `FLASHVSR_MODEL_PATH`.
- **Missing audio:** pass `--keep-audio`, check that the input has audio, and ensure both `ffmpeg` and `ffprobe` are installed.
- **NVENC issues:** hardware encoding requires a supported GPU and driver encoding libraries. For an initial check, run the FFmpeg command in [INSTALLATION.md](INSTALLATION.md#video-tools).

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
