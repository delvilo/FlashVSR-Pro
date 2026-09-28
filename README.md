# FlashVSR-Pro: Real-Time Video Super-Resolution

FlashVSR-Pro is an independent implementation of the diffusion-based streaming video super-resolution method from [FlashVSR](https://arxiv.org/abs/2510.12747). It provides a unified command-line interface for enhancing videos and image sequences.

FlashVSR-Pro supports direct Python execution on **native Linux and Google Colab**. See [INSTALLATION.md](INSTALLATION.md) for setup, model downloads, and troubleshooting.

## Features

- **Unified inference:** `flashvsr infer`, `batch`, and `long` share one validated configuration and engine; select `full`, `tiny`, or `tiny-long` mode.
- **Audio preservation:** `--keep-audio` transfers all input audio tracks to the processed video.
- **Lower VRAM use:** `--tile-dit` splits inference into overlapping tiles; `--tile-vae` enables tiled decoding in full mode.
- **Video alignment:** spatial and temporal padding accommodate model constraints, then output is cropped to the requested resolution and adjusted to the input frame count.
- **GPU acceleration:** CUDA inference, TF32/cuDNN optimizations, and high-quality HEVC NVENC encoding for MP4/MOV/MKV when available, with software encoding fallback.
- **Native installation:** `scripts/install.sh` installs the pinned Python dependencies and builds the bundled Block-Sparse-Attention backend.
- **Model management:** pinned versions, mode-specific downloads, SHA-256 verification and a shared cache.
- **Diagnostics:** leveled logs, optional JSONL logs, and automatic JSON reports with timings, FPS, peak VRAM, model identity and parameters.
- **Resumable long videos:** bounded frame buffers, lossless segment checkpoints and one model load per job invocation.

## Supported setup

| Component | Baseline |
| --- | --- |
| System | Linux; Google Colab with a compatible GPU runtime |
| Python | 3.13.x reference; 3.14.x compatibility target |
| PyTorch | 2.11.0+cu128 (CUDA 12.8 wheels) |
| CUDA Toolkit | 12.8 or newer (12.x); 12.8 matches the cu128 wheels |
| GPU | NVIDIA Ampere, Ada, or Hopper: for example RTX 3090/4090, A100, H100, or L4 |
| Build tools | C++17 compiler, Ninja, packaging, and wheel |
| Video tools | FFmpeg and FFprobe on `PATH` |

VRAM use depends on resolution, duration, mode, and tiling. Start with a short clip in `tiny` mode and `--tile-dit`; 16 GB or more gives more room for processing. A GPU being assigned in Colab does not guarantee compatibility: **T4 and P100 are unsupported** by this backend. Select A100 or L4 when available.

Windows, macOS, and WSL 2 are not supported. Project build scripts target Linux; bundled third-party source retains its upstream portability code.

## Linux quick start

Install an NVIDIA driver and the CUDA Toolkit using [NVIDIA's Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). The toolkit supplies `nvcc`; PyTorch wheels alone do not supply the compiler.

On a Linux distribution with Python 3.13 development and venv packages, install system dependencies and create a Python environment:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git git-lfs ffmpeg python3.13-venv python3.13-dev

git clone --depth 1 https://github.com/delvilo/FlashVSR-Pro.git
cd FlashVSR-Pro

# Use Python 3.13 (and its matching venv/dev packages) if the system Python differs.
python3.13 -m venv .venv
source .venv/bin/activate
bash scripts/install.sh
```

An existing Conda environment with a supported Python version can also be used. Activate it before invoking the installer; all subprocesses use the selected Python environment.
On distributions without Python 3.13 packages, install a matching interpreter and headers first or use a Python 3.13–3.14 Conda environment. CUDA 13.x toolkits are not supported with the pinned cu128 wheel; select a CUDA 12.8+ toolkit within the 12.x series.

Download and verify the model weights in the versioned cache:

```bash
python -m flashvsr models download --mode all
python -m flashvsr models check --mode all
```

Download and run a short sample:

```bash
python scripts/download_samples.py example0.mp4
python -m flashvsr infer -i inputs/example0.mp4 -o results/ --mode tiny --scale 4.0 --tile-dit
```

Sample videos are optional downloads. Use your own input directly, or see [inputs/README.md](inputs/README.md) to list and download other samples. Downloads are checked against a pinned size and SHA-256 manifest.

After opening a new shell, activate the same environment before running inference. Installation does not need to be repeated unless dependencies change.

## Google Colab

Open [colab/FlashVSR_Pro.ipynb](colab/FlashVSR_Pro.ipynb) in Google Colab, select an **A100 or L4 GPU runtime**, and run the cells in order. The notebook checks the GPU and toolkit, creates a dedicated Python environment in the checkout, downloads models, and runs `python -m flashvsr infer` as a subprocess. It downloads the default sample on demand; you can also select an uploaded video. Use a short input first.

The complete manual workflow is in [INSTALLATION.md — Google Colab](INSTALLATION.md#google-colab). Colab's runtime storage is temporary; download results or copy them to mounted Google Drive before the runtime is reset.

## Inference modes

| Mode | Decoder | Purpose |
| --- | --- | --- |
| `full` | Wan2.1 VAE | Quality-oriented processing with higher VRAM use |
| `tiny` | TCDecoder | Balanced speed, quality, and VRAM use; default |
| `tiny-long` | TCDecoder | Streaming model path for longer videos |

```bash
python scripts/download_samples.py example0.mp4 example4.mp4
python -m flashvsr infer -i inputs/example0.mp4 -o results/ --mode full --tile-vae
python -m flashvsr infer -i inputs/example0.mp4 -o results/ --mode tiny --keep-audio
python -m flashvsr infer -i inputs/example4.mp4 -o results/ --mode tiny-long --tile-dit
```

Use `flashvsr long` for bounded input memory and resumable processing. `infer` still prepares its entire input; choosing the `tiny-long` model alone does not enable job checkpoints.

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
| `--metrics-json` | Override automatic model, timing and VRAM report | `OUTPUT.json` |
| `--model-dir`, `--model-version` | Model location and version | Shared cache, `v1.1` |
| `--log-level`, `--log-file` | Log severity and optional JSONL file | `INFO`, stderr |

The model was designed for 4× upscaling; use `--scale 4.0` for that workflow. Run `python -m flashvsr infer --help` for the command-line reference. Sparse attention requires FP16/BF16 and a compatible CUDA GPU. Unsupported devices, precision, parameter values, and missing model files stop with an error; there is no CPU inference fallback.

An existing directory, or an output path without a suffix, is treated as an output directory. File paths such as `results/enhanced.mp4` select an explicit output filename. User-supplied input/output paths are relative to the current working directory. Model files use the shared versioned cache described below.

## Model management and diagnostics

```bash
python -m flashvsr models list --mode tiny
python -m flashvsr models download --mode tiny
python -m flashvsr models check --mode tiny
python -m flashvsr infer -i /data/input.mp4 -o /data/enhanced.mp4 \
  --mode tiny --keep-audio --log-level INFO --log-file results/run.jsonl
```

The default model directory is `~/.cache/flashvsr/v1.1` (or `$XDG_CACHE_HOME/flashvsr/v1.1`).
Set `FLASHVSR_CACHE_DIR` to choose another cache root. `--model-dir` or
`FLASHVSR_MODEL_PATH` selects an exact directory, including for existing weights:

```bash
python -m flashvsr models download --model-dir ./models/FlashVSR-v1.1
python -m flashvsr infer -i input.mp4 --model-dir ./models/FlashVSR-v1.1
```

The download command verifies and reuses valid weights, repairs corrupt files,
and copies the packaged fixed prompt into the same directory. No models are
downloaded implicitly during inference. The legacy `FLASHVSR-Pro_MODEL_PATH`
environment key remains readable. The pinned model manifest is
[`flashvsr/assets/models.json`](flashvsr/assets/models.json).

Each run writes `OUTPUT.json` with stage timings, inference/end-to-end FPS,
peak allocated/reserved VRAM, model revision/digests, package versions and all
parameters. `--metrics-json` overrides that path. Failure reports record the
stage and error. `--log-file` appends structured JSONL logs; `--log-level DEBUG`
adds diagnostic details and tracebacks.

## Batch and long-video programs

```bash
python -m flashvsr batch --input-dir inputs --output-dir results \
  --mode tiny --scale 2 --keep-audio --tile-dit --seed 42
python -m flashvsr long -i /data/long.mp4 -o /data/results \
  --segment-frames 129 --work-dir /data/jobs/long \
  --mode tiny --scale 2 --keep-audio --tile-dit
# After interruption, use the SAME settings and add --resume:
python -m flashvsr long -i /data/long.mp4 -o /data/results \
  --segment-frames 129 --work-dir /data/jobs/long \
  --mode tiny --scale 2 --keep-audio --tile-dit --resume
```

All three workflows accept the same mode, device, dtype, model, tiling, seed,
quality, color, FPS and audio options. Batch input/output defaults are relative
to the current working directory. Every eligible video is processed; there
are no filename-specific exclusions or automatic audio exceptions. The output
directory must be outside the input tree. A failed input is recorded and the
remaining inputs continue; any failure makes the final exit code 1.

Long jobs decode sequentially and keep at most `--segment-frames` input frames
(default 129) at once, regardless of keyframe spacing. `--segment-time` remains
an additional limit (default 60 seconds); the smaller limit determines each
segment. Model padding adds at most 24 repeated lookahead frames. Lower the frame
limit and use spatial tiling when a high resolution still exceeds VRAM.

Weights and fixed-prompt attention state are loaded once per invocation; video
and decoder caches are reset between segments. Completed outputs are stored as
lossless FFV1 files. The final encode streams from those files, using HEVC NVENC
with software fallback, and attaches every original audio track with `--keep-audio`.
The previous final video remains intact if encoding or verification fails.

`job.json` records input SHA-256, settings, model/runtime identity, exact frame
ranges and verified segment hashes. `--resume` rejects incompatible jobs, skips
valid segments and recomputes missing/corrupt segments. The default job directory
is `FINAL.mp4.job` next to the final video. A job lock prevents concurrent writers.
An existing job needs `--resume`; use a different `--work-dir` for changed settings.
Resuming may scan the decoded prefix, but does not infer it again.

Failed/cancelled jobs always retain completed segments. Success removes segment
files unless `--keep-temp` is set; the job record and final report remain. A
completed resume verifies the final file and returns without loading models.
Lossless checkpoints need disk space proportional to output duration. Keep the
input, output and job directory on persistent storage in Colab. Frame memory is
bounded by segment size, while job metadata grows with the number of segments.
Temporal state resets at each boundary, so visible seams remain possible. VFR
inputs retain their decoded frames and are normalized to the selected output FPS.

`infer.py`, `batch_inference.py` and `long_video_worker.py` remain thin
compatibility entry points. `--output_dir` and `--segment_time` aliases remain
accepted by the long-video command. Audio is opt-in with `--keep-audio` in all
workflows. Automatic output names include mode, scale and the source filename;
other parameters live in the JSON report. Use an explicit output file or a
separate output directory to retain runs with different parameters.

## Project files

| File or directory | Role |
| --- | --- |
| `flashvsr/cli.py`, `flashvsr/engine.py` | Unified CLI and inference core |
| `infer.py` | Compatibility entry point |
| `batch_inference.py` | Recursive batch processing |
| `long_video_worker.py` | Segment processing and concatenation |
| `scripts/install.sh` | Linux/Colab dependency and CUDA extension installation |
| `scripts/download_samples.py` | Optional sample downloads with size and SHA-256 verification |
| `inputs/samples.json`, `inputs/README.md` | Pinned sample manifest and download instructions |
| `colab/FlashVSR_Pro.ipynb` | Colab setup and inference notebook |
| `diffsynth/pipelines/flashvsr_*.py` | Three inference pipelines |
| `flashvsr/` | Configuration, workflows, media, models, logging and metrics |
| `utils/` | Decoders, projector, tiling, CUDA and checkpoint helpers |
| `Block-Sparse-Attention/` | Bundled sparse attention source and CUTLASS headers |
| `flashvsr/assets/` | Model manifest and bundled fixed prompt |
| `ARCHITECTURE.md` | Module boundaries, dependency trace, model and report contracts |
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
- **Missing model files:** run `python -m flashvsr models download --mode all`; use the same `--model-dir` or `FLASHVSR_MODEL_PATH` as inference.
- **Missing audio:** pass `--keep-audio`, check that the input has audio, and ensure both `ffmpeg` and `ffprobe` are installed.
- **NVENC issues:** the selected FFmpeg must complete a real HEVC test encode with the requested preset before hardware encoding is used. If hardware encoding or output validation fails, output is retried with libx264. See [INSTALLATION.md](INSTALLATION.md#video-tools).

## Reliability checks

The three command-line programs use exit codes **0** (success), **1** (runtime, input, model, or encoding failure), **2** (invalid arguments), and **130** (interrupted). `--help` works before installing the inference dependencies. Model loaders reject missing, empty, Git LFS pointer, corrupt, and incompatible weights instead of proceeding with uninitialized parameters.

One FFmpeg implementation handles encoding, audio muxing, and concatenation. With a compatible GPU, even-sized MP4/MOV/MKV outputs use HEVC NVENC preset `p7`, HQ tuning, full-resolution multipass VBR, three B-frames with middle references, 32-frame lookahead, spatial/temporal AQ, and `yuv420p`. Default `--quality 10` maps to `-cq 20`; lower quality settings raise CQ. AVI keeps H.264 NVENC, WebM uses VP9, and GIF uses GIF; missing or failed NVENC and odd dimensions use libx264 for compatible containers. Outputs are verified for codec, dimensions, frame count, FPS, and audio tracks before an atomic replacement; failed encodes preserve an existing output. `--keep-audio` preserves every audio track, re-encoding audio to AAC (Opus for WebM). Inputs without audio produce a silent output. Subtitles and attachments are not copied. GIF output cannot preserve audio.

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
