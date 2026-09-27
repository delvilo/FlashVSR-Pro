#!/usr/bin/env python3
"""Run real model inference in all three modes and write a GPU acceptance report."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from utils.cli import MODES, cuda_device, positive_float, run_cli
from utils.media import MediaTools, atomic_output
from utils.runtime import validate_cuda, validate_models


def write_report(path, report):
    with atomic_output(path) as temporary:
        temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'results/gpu-validation')
    parser.add_argument('--device', type=cuda_device, default='cuda')
    parser.add_argument('--max-vram-gib', type=positive_float, help='Optional per-mode peak reserved-memory ceiling')
    parser.add_argument('--timeout', type=positive_float, default=1800, help='Seconds allowed per mode')
    args = parser.parse_args(argv)
    directory = args.output_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Each run gets a new directory, so stale results cannot satisfy validation.
    run_dir = Path(tempfile.mkdtemp(prefix='run-', dir=directory))
    report_path = run_dir / 'report.json'
    report = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'python': sys.version, 'results': [], 'command': sys.argv}
    print(f'GPU acceptance report: {report_path}', flush=True)
    try:
        import torch
        index = validate_cuda(torch, args.device)
        for mode in MODES:
            validate_models(mode)
        properties = torch.cuda.get_device_properties(index)
        report.update({'gpu': properties.name, 'device': index, 'gpu_total_bytes': properties.total_memory,
                       'torch': torch.__version__, 'cuda': torch.version.cuda,
                       'max_vram_gib': args.max_vram_gib,
                       'packages': sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())})
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, capture_output=True, text=True)
        report['commit'] = revision.stdout.strip() if revision.returncode == 0 else None
        media = MediaTools()
        source = run_dir / 'input.mp4'
        media.run(['-f', 'lavfi', '-i', 'testsrc2=size=128x96:rate=8:duration=2.125',
                   '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2.125',
                   '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                   '-c:a', 'aac', source])
        report['input'] = media.verify_video(source, width=128, height=96, frames=17, audio_streams=1, fps=8)
        write_report(report_path, report)
        for mode in MODES:
            output = run_dir / f'{mode}.mp4'
            metrics_path = run_dir / f'{mode}.json'
            log_path = run_dir / f'{mode}.log'
            command = [sys.executable, str(PROJECT / 'infer.py'), '-i', str(source), '-o', str(output),
                       '--mode', mode, '--scale', '2.0', '--device', args.device, '--keep-audio',
                       '--tile-dit', '--metrics-json', str(metrics_path)]
            if mode == 'full':
                command.append('--tile-vae')
            result = {'mode': mode, 'status': 'running', 'command': command, 'log': str(log_path)}
            report['results'].append(result)
            write_report(report_path, report)
            with log_path.open('w') as log:
                process = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            if process.returncode:
                raise RuntimeError(f'{mode} exited {process.returncode}; see {log_path}')
            result['output'] = media.verify_video(output, width=256, height=192, frames=17, audio_streams=1, fps=8)
            metrics = json.loads(metrics_path.read_text())
            allocated, reserved = metrics['peak_allocated_bytes'], metrics['peak_reserved_bytes']
            if not 0 < allocated <= reserved <= properties.total_memory:
                raise RuntimeError(f'{mode} reported invalid GPU memory measurements: {allocated}, {reserved}')
            if args.max_vram_gib is not None and reserved > args.max_vram_gib * 1024**3:
                raise RuntimeError(f'{mode} exceeded --max-vram-gib: {reserved / 1024**3:.2f} GiB')
            result.update({'status': 'passed', 'metrics': metrics})
            write_report(report_path, report)
        report['status'] = 'passed'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        if report['results'] and report['results'][-1]['status'] == 'running':
            report['results'][-1]['status'] = 'failed'
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        write_report(report_path, report)
    print(f'All three modes passed: {report_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(run_cli(_main))
