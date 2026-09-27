# Testing and acceptance

The CPU suite checks installation metadata, arguments, missing or corrupt weights, launcher exit codes, and real FFmpeg output without requiring the large model downloads. It does not execute CUDA kernels or measure inference quality.

## CPU environment

From a checkout using Linux and Python 3.12–3.14:

```bash
python3 -m venv .venv-test
source .venv-test/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-test-torch.txt
# Install FFmpeg/FFprobe with the system package manager if absent.
python -m unittest discover -s tests -v
python -m build --no-isolation
python scripts/check_packages.py
```

CPU PyTorch wheels are for these tests only. Create a separate environment for CUDA inference and use `scripts/install.sh`.

The tests cover safe sample downloads, invalid CLI parameters, help before dependency installation, missing/empty/LFS-pointer/corrupt/incompatible weights, unsupported CUDA environments, installer failure before package changes, and consistent exit codes. FFmpeg tests verify actual software encoding, multiple audio tracks, frame counts, quoted paths, odd dimensions, and preservation of existing output when an encode or validation fails. Hardware failures are injected to exercise the NVENC retry path on CPU machines.

`check_packages.py` inspects the built wheel and source archive, checking package metadata, tokenizer data, the prompt, CUDA sources, and licenses. The application still runs from a source checkout/archive with the separately compiled backend; see [INSTALLATION.md](INSTALLATION.md#package-and-development-layout).

The `Linux CPU checks` workflow runs the tests, builds both distributions, installs application dependencies with CPU wheels, and runs `pip check` on Python 3.12, 3.13 and 3.14. The installer is not fully executed on a CPU runner; its preflight failures are tested. CUDA extension compilation belongs to GPU acceptance.

## Real GPU acceptance

Use native Linux or the provided Colab notebook with an Ampere/Ada/Hopper GPU, CUDA Toolkit 12.5+ (12.x), the installed backend, and all model weights. The reference Python version is 3.12; Python 3.13 and 3.14 are included in the CPU CI. Toolkit 12.5 with cu126 wheels produces a minor-version warning during extension compilation; GPU acceptance is needed to verify that combination on the actual hardware.

```bash
bash scripts/install.sh
python scripts/validate_gpu.py
# Optional memory ceiling (choose one appropriate for your GPU):
python scripts/validate_gpu.py --max-vram-gib 22
```

Each run creates a new directory under `results/gpu-validation/` and generates its own 128×96, 17-frame, 8 FPS input with one audio track. It launches **full**, **tiny**, and **tiny-long** in separate subprocesses, with 2× scaling and tiled DiT; full also enables tiled VAE decoding. All weights, including the full-mode VAE, must exist before the run starts.

A passing run requires all three outputs to have **256×192 pixels, 17 frames, 8 FPS, and one audio track**. It also requires nonzero, consistent CUDA peak allocated/reserved memory measurements within the GPU's capacity and any supplied ceiling. Each mode has a timeout (default 1,800 seconds) and writes a log and metrics file. `report.json` records the commit when available, Python and package versions, GPU, CUDA build, commands, output properties, and memory/timing metrics. Failure or interruption marks the report failed and returns a nonzero status. Existing reports cannot satisfy a new run.

This is a correctness and memory baseline for a small synthetic input, not a visual-quality benchmark or a guarantee that longer/higher-resolution videos fit in VRAM. Compare runs on the same GPU, toolkit, weights, mode settings, and input before drawing performance conclusions. Metrics include model loading and preprocessing in peak memory; inference timing is synchronized with CUDA.

The Colab notebook has an optional final acceptance cell. The manually dispatched `GPU acceptance` workflow uses a runner labeled `self-hosted`, `linux`, `x64`, and `flashvsr-gpu`. Configure that runner's Python, CUDA 12.5+ toolkit (12.x), and `FLASHVSR_MODEL_PATH` pointing to downloaded weights outside the checkout. The workflow installs/builds the project and uploads reports and logs even if validation fails. GPU jobs do not run automatically for pull requests.

A CPU test pass or a supplied GPU script is **not a GPU acceptance result**. Publish the generated report only after running it on actual supported hardware.
