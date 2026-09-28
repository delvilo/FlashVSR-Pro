# Colab workflow

Open [colab/FlashVSR_Pro.ipynb](colab/FlashVSR_Pro.ipynb) in Google Colab and run the cells in order. This workflow uses **A100 or L4**, Ubuntu **24.04 noble**, native Python **3.13**, PyTorch **2.11.0+cu128**, and CUDA Toolkit **12.8**. T4/P100, older Colab images and alternate Python environments are unsupported by this notebook.

The CUDA version shown by `nvidia-smi` describes driver compatibility. A driver reporting CUDA 13.0 is acceptable with the installed 12.8 toolkit and cu128 PyTorch; the workflow checks `nvcc` separately. Native Linux retains the broader versions documented in [INSTALLATION.md](INSTALLATION.md).

## Start a job

1. Select an **A100 or L4 GPU runtime**. Mount Drive in the first cell and choose the code revision, paths and mode.
2. Run diagnostics. They check the OS, Python headers, PyTorch, GPU/VRAM, toolkit, compiler, FFmpeg/FFprobe, a real HEVC NVENC encode and storage. An A100 can use software encoding while inference runs on its GPU.
3. Run setup. It keeps Colab's matching PyTorch, installs the pinned application/build dependencies, restores or builds the attention wheel, executes a CUDA correctness check, and prepares only the selected mode's models.
4. Put the input in `MyDrive/FlashVSR-Pro/inputs/`, or supply another video path. Inputs outside the managed Drive root are copied into the job before inference so they survive a VM reset.
5. Choose the video settings and run. Record the generated job name. The default is `tiny`, 2× scaling, 129 frames per segment, DiT tiling, BF16 and audio preservation; full mode also tiles its VAE. Lower the segment size if VRAM is insufficient.
6. Retrieve `results/JOB/enhanced.mp4` from Drive. Browser download is optional. The accompanying reports include parameters, code/model versions, timings, FPS and peak VRAM.

Use `main` for new jobs after this workflow is merged. While testing the unmerged change, set the notebook's `project_ref` to `codex/colab-workflow`. To resume later, use the exact commit recorded by that job rather than a newer branch tip.

The notebook uses the native interpreter and fresh subprocesses for setup and inference. It does not create a virtual environment or reinstall the large PyTorch CUDA wheels. Package changes do not require restarting the notebook kernel because inference imports packages in a new process. Avoid running unrelated package installers in the same runtime.

## Paths and the 20 GB budget

| Setting | Default | Purpose |
| --- | --- | --- |
| Drive root | `/content/drive/MyDrive/FlashVSR-Pro` | Persistent models, wheels, inputs, jobs and results |
| Local root | `/content/flashvsr` | Local models, build workspace, video processing and logs |
| Checkout | `/content/FlashVSR-Pro` | Selected Git revision |
| Drive budget | 20 GB | Maximum managed directory usage, including temporary copies and reserve |
| Reserve | 2 GB | Space kept available inside that budget |

GB means 1,000,000,000 bytes. The default leaves at most **18 GB of managed files**, counting existing inputs/results and temporary replacement files. Model weights use approximately 6.44 GB for `tiny`/`tiny-long` or 6.95 GB for all modes; compiled wheels, videos and checkpoints need additional space. This is a project budget, not a Google account quota query: the mount's reported free space can differ from the actual account allowance.

Inference reads local files. After each segment, the lossless output is copied to Drive, verified by SHA-256 read-back, and only then added to the saved checkpoint record. Lossless FFV1 files can be much larger than the source video, so **20 GB cannot guarantee completion of arbitrary long/high-resolution videos**. Final publication also needs room for the final video before segment deletion. Reducing segment length reduces working memory; it does not substantially reduce the accumulated lossless storage.

If space or Drive I/O runs out, the operation stops and preserves completed checkpoints. Free space from other finished jobs or old caches, then resume. An unsaved local segment is retained while that VM still exists; a reset may require recomputing it. Input staging and final publication use verified temporary copies, preserving the previous destination on a failed copy. Read-back validates the mounted copy; it cannot guarantee Google's server-side persistence after a sudden mount/runtime loss.

Only one active notebook may write to a Drive root. A local lock prevents overlapping cells in one VM; it is not a distributed lock between different Colab runtimes. Use separate roots for concurrent notebooks.

## Resume after interruption or runtime reset

1. Mount the same Drive directory. Inspect **saved jobs and storage** for the job name, `code_commit`, runtime and parameters.
2. Set `project_ref` to the saved commit and rerun the checkout, diagnostics and setup cells with the same mode. Reuse the same local-root path. A matching model/wheel cache avoids network downloads and compilation, while integrity and CUDA checks still run.
3. Set the saved job name, enable `resume_job`, and restore the original scale, segment size, tiling, seed, color and audio settings. The input field can remain empty on resume because the source reference is saved.
4. Run again. Valid saved segments are restored and skipped; missing/corrupt segments are recomputed. A completed job verifies its saved results and returns without loading a model.

Resume rejects changed code, Python/package versions, model identity, input content or processing settings. If a newer Colab image changes pinned runtime versions, create a new job; the workflow does not recreate older Colab images. Do not edit or delete referenced inputs while a job is unfinished. Independent segments reset streaming/decoder state, so inspect temporal boundaries on representative footage.

## Persistent files and cleanup

| Drive path | Contents / retention |
| --- | --- |
| `models/v1.1/` | Pinned models and fixed prompt; validated by size and SHA-256 |
| `wheels/KEY/` | Compiled wheel and manifest; retained for the matching build environment |
| `inputs/` | User-managed source videos; referenced without an extra persistent copy |
| `jobs/JOB/` | Input copy when needed, settings, progress, failure logs and completion record |
| `jobs/JOB/checkpoint/segments/` | Verified lossless segments; retained on failure |
| `results/JOB/` | `enhanced.mp4`, `report.json`, `run.jsonl`, `parameters.json`, `environment.json` |
| `diagnostics/` | Latest preflight/setup reports and setup log; GPU acceptance evidence in separate run directories |

The completion record is published only after every result file has been saved and verified. Segments are then removed unless **keep_segments_after_success** is enabled. Inputs, old results and model/wheel caches are never automatically evicted. Delete finished jobs/results and unused caches manually when no longer needed; retained files in Drive Trash can still affect quota. Keep the job record if you want completed-result verification through resume. Never delete an unfinished job's checkpoint directory.

Wheel keys include Python ABI, PyTorch/CUDA versions, GPU capability, compiler, libc, bundled source content and build requirements. Changing any of these selects another cache. Both cache hits and source builds must pass a sparse CUDA computation compared with PyTorch SDPA; an import alone is insufficient. A failed cached wheel gets one source rebuild. Builds use two workers and the `sm_80` target for the A100/L4 baseline; a switch between A100 and L4 creates a separate cache key and still requires validation.

## Commands and GPU acceptance

The notebook invokes these commands using its `sys.executable`, from the checkout:

```bash
python -m flashvsr.colab diagnose
python -m flashvsr.colab setup --mode tiny
python -m flashvsr.colab status
python -m flashvsr.colab run --job example --input /content/drive/MyDrive/FlashVSR-Pro/inputs/input.mp4
# Same settings, after interruption:
python -m flashvsr.colab run --job example --resume
```

Each command accepts `--drive-root`, `--local-root`, `--drive-budget-gb`, `--reserve-gb`, and `--project`. Use `run --help` for video parameters. Errors return 1, usage errors 2, and cancellation 130.

The optional final notebook cell prepares all model modes and runs:

```bash
python -m flashvsr.colab setup --mode all
python -m flashvsr.colab validate
```

Acceptance runs locally, then saves its videos, logs and reports under `diagnostics/gpu-validation/` using the same budget and verification rules, including failure evidence when space is available. It checks all three modes, segmented processing, model reuse, completed-job resume, resolution, frame count, audio and peak VRAM. See [TESTING.md](TESTING.md). Run it separately on an actual A100 and L4 before claiming GPU acceptance; CPU CI does not validate CUDA compilation or kernel execution.
