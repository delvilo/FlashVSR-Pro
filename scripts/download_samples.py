#!/usr/bin/env python3
"""Download optional sample videos from a pinned, checksum-verified snapshot."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from flashvsr.observability import configure_logging

logger = logging.getLogger(__name__)
MANIFEST_PATH = PROJECT_DIR / "inputs/samples.json"


def load_manifest(path=MANIFEST_PATH):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("Unsupported sample manifest version")
    for name, sample in manifest["samples"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
            raise ValueError(f"Invalid sample path: {name}")
        if sample["bytes"] <= 0 or len(sample["sha256"]) != 64:
            raise ValueError(f"Invalid sample metadata: {name}")
    return manifest


def matches_sample(path, sample):
    if not path.is_file() or path.stat().st_size != sample["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == sample["sha256"]


def download_sample(base_url, name, sample, output_dir, force=False):
    destination = Path(output_dir) / name
    if matches_sample(destination, sample):
        logger.info("Already verified: %s", destination)
        return destination
    if destination.exists() and not force:
        raise FileExistsError(
            f"Existing file differs from the sample: {destination}. "
            "Choose another --output-dir or use --force to replace it."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = base_url.rstrip("/") + "/" + quote(name, safe="/")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=destination.name + ".", suffix=".part", delete=False
        ) as target:
            temporary = Path(target.name)
            digest = hashlib.sha256()
            received = 0
            with urlopen(url, timeout=60) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    received += len(chunk)
                    if received > sample["bytes"]:
                        raise ValueError(f"Downloaded size exceeds manifest for {name}")
                    digest.update(chunk)
                    target.write(chunk)
            if received != sample["bytes"] or digest.hexdigest() != sample["sha256"]:
                raise ValueError(f"Size or SHA-256 verification failed for {name}")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    logger.info("Downloaded and verified: %s", destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", nargs="*", help="Sample paths from --list (default: example0.mp4)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--list", action="store_true", help="List samples without downloading")
    selection.add_argument("--all", action="store_true", help="Download the entire optional collection")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "inputs")
    parser.add_argument("--force", action="store_true", help="Replace a mismatched local file after verification")
    args = parser.parse_args(argv)
    configure_logging()
    if args.samples and (args.all or args.list):
        parser.error("Choose sample names, --all, or --list")

    try:
        manifest = load_manifest()
        samples = manifest["samples"]
        if args.list:
            for name, sample in samples.items():
                print(f"{sample['bytes'] / 1_000_000:7.2f} MB  {name}")
            return 0
        names = list(samples) if args.all else (args.samples or ["example0.mp4"])
        unknown = [name for name in names if name not in samples]
        if unknown:
            parser.error(f"Unknown sample(s): {', '.join(unknown)}. Use --list.")
        output_dir = args.output_dir.expanduser().resolve()
        for name in dict.fromkeys(names):
            download_sample(manifest["base_url"], name, samples[name], output_dir, args.force)
    except (OSError, URLError, ValueError) as error:
        logger.error("Sample download failed: %s", error)
        return 1
    except KeyboardInterrupt:
        logger.warning("Sample download cancelled")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
