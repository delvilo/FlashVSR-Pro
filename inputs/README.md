# Optional sample videos

The checkout contains the sample manifest, not the video binaries. From the project directory, use Python 3.10–3.12 to download only what you need:

```bash
# Default: example0.mp4
python scripts/download_samples.py

# List all paths and sizes without downloading
python scripts/download_samples.py --list

# Download selected samples (subdirectories are preserved)
python scripts/download_samples.py example0.mp4 example4.mp4 JIUTIAN-gen/test_t2v0.mp4

# Download the full collection: 78 videos, approximately 178 MB
python scripts/download_samples.py --all
```

No inference dependencies or model weights are needed for this downloader. Run inference separately after installing the project and downloading the weights.

By default, files are placed in this checkout's `inputs/` directory, even when the script is called from another working directory. To use another location:

```bash
python scripts/download_samples.py example0.mp4 --output-dir /data/flashvsr-samples
```

The [manifest](samples.json) records each sample's path, byte count, and SHA-256. Downloads come from the immutable repository snapshot [`4fa8c51`](https://github.com/delvilo/FlashVSR-Pro/tree/4fa8c51af08174379418e28550c903367bdc9662/inputs), which preserves the original collection. Downloaded samples are ignored by Git.

Already verified files are reused without network access. A download is written to a temporary file and moved into place only after its size and hash match. A mismatched existing file is preserved by default; use `--force` to replace it after successful verification, or choose a different output directory. Failed downloads are discarded and can be retried with the same command.

Your own videos can be placed here or passed directly to `infer.py` with any input path. `batch_inference.py` processes videos under this directory; it ignores the manifest and this README. No sample download is required to process your own inputs.
