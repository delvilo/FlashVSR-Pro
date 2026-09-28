# Testing and acceptance

The CPU suite checks installation metadata, arguments, missing or corrupt weights, launcher exit codes, and real FFmpeg output without requiring the large model downloads. It does not execute CUDA kernels or measure inference quality.

## CPU environment

From a checkout using Linux and Python 3.13–3.14:

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

The tests cover safe sample downloads, invalid CLI parameters, help before dependency installation, missing/empty/LFS-pointer/corrupt/incompatible weights, unsupported CUDA environments, installer failure before package changes, and consistent exit codes. FFmpeg tests verify actual software encoding, multiple audio tracks, frame counts, quoted paths, odd dimensions, and preservation of existing output when an encode or validation fails. HEVC NVENC preset selection, its full-settings probe, and hardware or codec-verification failures are simulated on CPU runners; real NVENC encoding requires a supported GPU.

`check_packages.py` inspects the built wheel and source archive, checking package metadata, the model manifest, packaged prompt, CLI, CUDA sources, licenses, and absence of tokenizer data. The installed CLI also works outside the checkout, with the separately compiled backend; see [INSTALLATION.md](INSTALLATION.md#package-and-development-layout).

The `Linux CPU checks` workflow runs the tests, builds both distributions, installs application dependencies with CPU wheels, and runs `pip check` on Python 3.13 and 3.14. The installer is not fully executed on a CPU runner; its preflight failures are tested. CUDA extension compilation belongs to GPU acceptance.

## Real GPU acceptance

Use native Linux with an Ampere/Ada/Hopper GPU, CUDA Toolkit 12.8+ (12.x), the installed backend, and all model weights. The Colab notebook specifically targets A100/L4, Ubuntu 24.04, native Python 3.13, PyTorch 2.11.0+cu128 and Toolkit 12.8. Python 3.13 and 3.14 are included in the CPU CI. Toolkit 12.9 with cu128 wheels may produce a minor-version warning during Linux extension compilation; GPU acceptance is needed to verify that combination on the actual hardware.

```bash
bash scripts/install.sh
python -m flashvsr models download --mode all
python scripts/validate_gpu.py
# Optional memory ceiling (choose one appropriate for your GPU):
python scripts/validate_gpu.py --max-vram-gib 22
# Include two bounded segments, one model load and completed-job resume per mode:
python scripts/validate_gpu.py --include-long
```

Each run creates a new directory under `results/gpu-validation/` and generates its own 128×96, 17-frame, 8 FPS input with one audio track. It launches **full**, **tiny**, and **tiny-long** in separate subprocesses, with 2× scaling and tiled DiT; full also enables tiled VAE decoding. All weights, including the full-mode VAE, must exist before the run starts.

A passing run requires all three outputs to have **256×192 pixels, 17 frames, 8 FPS, and one audio track**. It also requires nonzero, consistent CUDA peak allocated/reserved memory measurements within the GPU's capacity and any supplied ceiling. Each mode has a timeout (default 1,800 seconds) and writes a log and metrics file. `report.json` records the commit when available, Python and package versions, GPU, CUDA build, commands, output properties (including `video_codec`), and memory/timing metrics. A GPU acceptance pass can use H.264 software fallback; check for `video_codec: hevc` in each output to confirm HEVC NVENC was exercised on that machine. Failure or interruption marks the report failed and returns a nonzero status. Existing reports cannot satisfy a new run.

This is a correctness and memory baseline for a small synthetic input, not a visual-quality benchmark or a guarantee that longer/higher-resolution videos fit in VRAM. Compare runs on the same GPU, toolkit, weights, mode settings, and input before drawing performance conclusions. Metrics include model loading and preprocessing in peak memory; inference timing is synchronized with CUDA.

The Colab notebook has an optional final acceptance cell, using `python -m flashvsr.colab setup --mode all` and `python -m flashvsr.colab validate`. Computation runs locally; evidence, including failure logs when space permits, is copied and verified within the same Drive budget. Execute it on an actual A100 and L4 separately. Also interrupt a representative notebook job after a saved segment, reset the VM, rerun setup, and resume with the saved code/settings. Confirm that the compiled wheel is reused, only remaining segments infer, and results/audio are preserved. Local filesystem simulations cannot establish Google Drive mount durability or Colab image compatibility.

The manually dispatched `GPU acceptance` workflow uses a runner labeled `self-hosted`, `linux`, `x64`, and `flashvsr-gpu`. Configure that runner's Python 3.13 or 3.14, CUDA 12.8+ toolkit (12.x), and `FLASHVSR_MODEL_PATH` pointing to verified weights (including `posi_prompt.pth`) outside the checkout. Run `flashvsr models download --mode all` once using that directory. The workflow installs/builds the project and uploads reports and logs even if validation fails. GPU jobs do not run automatically for pull requests.

A CPU test pass or a supplied GPU script is **not a GPU acceptance result**. Publish the generated report only after running it on actual supported hardware.

`--include-long` also processes the fixture as 9+8-frame segments for every mode,
checks resolution/frame count/audio and each segment's memory, requires one model
load and a reused second call, then resumes the completed job and requires zero
model loads. The self-hosted GPU workflow enables this check. Inspect temporal
boundaries on representative footage separately; the small fixture cannot assess seams.

## Architecture regression coverage

Colab tests use real temporary files and FFmpeg with a CPU inference double. They simulate a destroyed local runtime, resume from Drive checkpoints, corrupt models/wheels/segments, failed segment/final uploads, retry after interrupted job registration, storage-budget exhaustion, unsupported runtime/GPU rejection, build-cache identity changes, cached-kernel failure/rebuild, and failed GPU validation report preservation. Notebook code cells are syntax-checked. CUDA builds and operations are mocked here and must pass the separate GPU gate above.

The suite additionally covers shared CLI configuration across all workflows,
mode-specific model selection, cache precedence, pinned revision URLs, missing,
truncated and same-size corrupt weights, atomic download interruption/cleanup,
and copying the packaged prompt without a network request. It checks failure
reports, model provenance, phase timings, FPS, memory fields and resource cleanup
using CPU test doubles. Real FFmpeg tests exercise long-video concatenation and
preserve multiple original audio tracks, as well as retaining previous output
when a segment fails.

Long-job tests use real FFmpeg with a long-GOP source and a five-frame buffer
ceiling, including a short final segment. They verify lossless pixel order,
buffer release, cancellation/resume, cached-segment corruption, changed settings,
exclusive job locks, final encoding failure/retry, NVENC software fallback and
completed jobs requiring no model load. CPU model doubles verify that a session
loads once, clears per-video state, evicts failed models and closes on interruption.
These tests do not establish visual continuity or GPU memory behavior at boundaries.

A subprocess imports all three retained pipelines with only the compiled
attention function stubbed, and instantiates the manifest DiT architecture on
PyTorch's meta device to verify its tensor key/shape fingerprint. Static import
checks reject wildcard imports and removed dependencies. These tests detect
packaging/orchestration regressions, not GPU inference quality or kernel safety.

CI builds the wheel and installs it into a fresh isolated environment, then
runs the installed model listing command outside the checkout. GPU acceptance
continues to require all three modes on supported hardware and validates the
new model/version/performance report fields. No CPU result substitutes for that gate.
