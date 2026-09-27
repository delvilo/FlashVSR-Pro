# Bundled dependency sources

FlashVSR-Pro targets native Linux and Google Colab. Its attention backend and CUTLASS are vendored as regular source files, so a checkout or source archive includes the files needed to build the CUDA extension without initializing Git submodules.

| Directory | Upstream | License |
| --- | --- | --- |
| `Block-Sparse-Attention/` | [LujiaJin/Block-Sparse-Attention](https://github.com/LujiaJin/Block-Sparse-Attention), originally selected via `flashvsr-fix` | [BSD 3-Clause](Block-Sparse-Attention/LICENSE) |
| `Block-Sparse-Attention/csrc/cutlass/` | [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass); bundled CMake metadata declares version 3.3.0 | [BSD 3-Clause](Block-Sparse-Attention/csrc/cutlass/LICENSE.txt) |

The original import did not record exact upstream commit IDs. For a reproducible reference, the bundled sources before this cleanup are preserved in this repository at [`4fa8c51`](https://github.com/delvilo/FlashVSR-Pro/tree/4fa8c51af08174379418e28550c903367bdc9662/Block-Sparse-Attention). Do not treat the upstream branch name or version label as proof of an exact upstream revision.

## Retained build sources

`Block-Sparse-Attention/setup.py` compiles the backend's C++/CUDA sources and includes `csrc/cutlass/include`. The cleanup retains those sources, headers, and license files. The source-distribution manifest also includes the CUTLASS license. Missing bundled headers now produce an explicit error asking for a complete source checkout or source distribution.

Project build scripts use Linux platform tags and compiler flags. Unmodified CUTLASS code, examples, tests, and documentation sources can still contain platform conditionals inherited from upstream; these do not establish Windows, macOS, or WSL support for FlashVSR-Pro.

## Generated documentation

The generated directories `csrc/cutlass/docs/` and `csrc/cutlass/python/docs/` were removed from version control. They contain rendered HTML, JavaScript, images, and search indexes; they are not inputs to the FlashVSR-Pro extension build or inference pipeline. Documentation sources remain in the vendored tree.

Use the [upstream CUTLASS repository](https://github.com/NVIDIA/cutlass) for current documentation. The [archived generated C++ documentation](https://github.com/delvilo/FlashVSR-Pro/tree/4fa8c51af08174379418e28550c903367bdc9662/Block-Sparse-Attention/csrc/cutlass/docs) and [Python documentation](https://github.com/delvilo/FlashVSR-Pro/tree/4fa8c51af08174379418e28550c903367bdc9662/Block-Sparse-Attention/csrc/cutlass/python/docs) remain available in the earlier snapshot.

The CUTLASS directory is marked `linguist-vendored` in `.gitattributes` so GitHub language statistics focus on project code. This affects classification only; retained dependency files are still tracked and distributed.

## Maintenance

The old `.gitmodules` declarations were removed because these directories are ordinary tracked files, not Git submodule entries. Update bundled dependencies deliberately, preserve their licenses and local changes, and verify a Linux CUDA build after changing them. Upstream documentation links inside the retained vendor sources may point to generated pages that are no longer present locally.

## Minimal DiffSynth subset

`diffsynth/` retains only FlashVSR pipelines, their shared color/base code, Wan
DiT/VAE, the flow-match scheduler and VRAM helpers. Unrelated models, pipelines,
trainers, prompt/tokenizer data and extensions were removed after tracing the
runtime dependency closure. The repository's Apache-2.0 license and FlashVSR
credits remain. See [ARCHITECTURE.md](ARCHITECTURE.md) for the dependency map and
pinned upstream model metadata. CUDA and CUTLASS algorithms are unchanged by
this application refactor.
