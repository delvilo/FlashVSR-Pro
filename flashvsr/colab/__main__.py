"""Notebook cells call fresh Python processes to avoid stale imported packages."""

import argparse
import json
import logging
from pathlib import Path
import sys

from flashvsr.config import InferenceConfig, MODES
from flashvsr.observability import configure_logging, write_report
from .storage import DriveStore, local_lock


def parser():
    root = argparse.ArgumentParser(description="FlashVSR Colab: Ubuntu 24.04, Python 3.13, A100/L4")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("diagnose", "setup", "run", "status", "validate"):
        action = sub.add_parser(name)
        action.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive/FlashVSR-Pro"))
        action.add_argument("--local-root", type=Path, default=Path("/content/flashvsr"))
        action.add_argument("--drive-budget-gb", type=float, default=20)
        action.add_argument("--reserve-gb", type=float, default=2)
        action.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[2])
        if name in ("setup", "run"):
            action.add_argument("--mode", choices=("all", *MODES) if name == "setup" else MODES, default="tiny")
        if name == "run":
            action.add_argument("--input", type=Path)
            action.add_argument("--job", required=True)
            action.add_argument("--resume", action="store_true")
            action.add_argument("--scale", type=float, default=2)
            action.add_argument("--segment-frames", type=int, default=129)
            action.add_argument("--tile-size", type=int, default=256)
            action.add_argument("--overlap", type=int, default=24)
            action.add_argument("--seed", type=int, default=0)
            action.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
            action.add_argument("--fps", type=float)
            action.add_argument("--quality", type=int, default=10)
            action.add_argument("--color-fix", action="store_true")
            action.add_argument("--no-audio", action="store_true")
            action.add_argument("--keep-temp", action="store_true")
    sub.add_parser("smoke", help="Internal: test the installed attention wheel on the GPU")
    sub.add_parser("check-dependencies", help="Internal: validate the project's dependency closure")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    configure_logging()
    try:
        from .setup import cuda_smoke, diagnose, prepare, validate_baseline, verify_requirements
        if args.command == "smoke":
            cuda_smoke()
            print("Sparse CUDA computation verified")
            return 0
        if args.command == "check-dependencies":
            print(json.dumps(verify_requirements(Path(__file__).resolve().parents[2]), indent=2))
            return 0
        local = args.local_root.expanduser().resolve()
        drive = args.drive_root.expanduser().resolve()
        if local == drive or local in drive.parents or drive in local.parents:
            raise ValueError("Local work and Drive storage must be separate directories")
        # Do not silently create a fake local /content/drive when not mounted.
        mount = Path("/content/drive")
        if drive.is_relative_to(mount) and not (mount / "MyDrive").is_dir() and not (mount / "Shareddrives").is_dir():
            raise RuntimeError("Mount Google Drive before using the notebook workflow")
        local.mkdir(parents=True, exist_ok=True)
        store = DriveStore(drive, args.drive_budget_gb, args.reserve_gb)
        if args.command == "status":
            from .workflow import list_jobs
            data = {"storage": store.capacity(), "jobs": list_jobs(store)}
        elif args.command == "diagnose":
            with local_lock(local / ".colab.lock"):
                data = diagnose(args.project, store, local)
                write_report(local / "diagnostics.json", data)
                store.copy(local / "diagnostics.json", "diagnostics/preflight.json")
        elif args.command == "setup":
            data = prepare(args.project, store, local, args.mode)
        elif args.command == "validate":
            import torch
            validate_baseline(torch)
            from .workflow import validate_gpu
            data = validate_gpu(args.project, store, local)
        else:
            import torch
            validate_baseline(torch)
            if args.segment_frames <= 0:
                raise ValueError("--segment-frames must be positive")
            from .workflow import run_job
            config = InferenceConfig(mode=args.mode, scale=args.scale, seed=args.seed, dtype=args.dtype,
                                     tile_dit=True, tile_vae=args.mode == "full", tile_size=args.tile_size,
                                     overlap=args.overlap, keep_audio=not args.no_audio,
                                     quality=args.quality, fps=args.fps, color_fix=args.color_fix)
            data = run_job(args.project, store, local, args.job, config, args.input,
                           resume=args.resume, segment_frames=args.segment_frames, keep_temp=args.keep_temp)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        logging.getLogger(__name__).error("Cancelled; completed Drive checkpoints remain available")
        return 130
    except Exception as error:
        logging.getLogger(__name__).error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
