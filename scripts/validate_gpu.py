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
from flashvsr.config import MODES
from flashvsr.cli import positive_float, run_cli
from flashvsr.observability import configure_logging
from utils.media import MediaTools, atomic_output
from utils.runtime import validate_cuda, validate_models


def write_report(path, report):
    with atomic_output(path) as temporary:
        temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'results/gpu-validation')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-vram-gib', type=positive_float, help='Optional per-mode peak reserved-memory ceiling')
    parser.add_argument('--timeout', type=positive_float, default=1800, help='Seconds allowed per mode')
    parser.add_argument('--include-long', action='store_true', help='Also validate bounded segments, model reuse and completed-job resume in every mode')
    args = parser.parse_args(argv)
    directory = args.output_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Each run gets a new directory, so stale results cannot satisfy validation.
    run_dir = Path(tempfile.mkdtemp(prefix='run-', dir=directory))
    report_path = run_dir / 'report.json'
    report = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'python': sys.version, 'results': [], 'command': sys.argv}
    configure_logging()
    import logging
    logging.info('GPU acceptance report: %s', report_path)
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
            if metrics['status'] != 'ok' or metrics['inference_fps'] <= 0 or not metrics['models']['revision']:
                raise RuntimeError(f'{mode} has incomplete metrics or model provenance')
            allocated, reserved = metrics['peak_allocated_bytes'], metrics['peak_reserved_bytes']
            if not 0 < allocated <= reserved <= properties.total_memory:
                raise RuntimeError(f'{mode} reported invalid GPU memory measurements: {allocated}, {reserved}')
            if args.max_vram_gib is not None and reserved > args.max_vram_gib * 1024**3:
                raise RuntimeError(f'{mode} exceeded --max-vram-gib: {reserved / 1024**3:.2f} GiB')
            result['metrics'] = metrics
            if args.include_long:
                folder = run_dir / f'{mode}-long'
                long_metrics = run_dir / f'{mode}-long.json'
                long_log = run_dir / f'{mode}-long.log'
                long_command = [sys.executable, '-m', 'flashvsr', 'long', '-i', str(source), '-o', str(folder),
                                '--mode', mode, '--scale', '2.0', '--device', args.device, '--keep-audio',
                                '--tile-dit', '--segment-frames', '9', '--keep-temp', '--metrics-json', str(long_metrics)]
                if mode == 'full':
                    long_command.append('--tile-vae')
                with long_log.open('w') as log:
                    subprocess.run(long_command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=args.timeout, check=True)
                long_result = json.loads(long_metrics.read_text())
                media.verify_video(folder / 'FlashVSR_input_Final.mp4', width=256, height=192,
                                   frames=17, fps=8, audio_streams=1)
                if (long_result['model_loads'] != 1 or long_result['segments_processed'] != 2 or
                        [item['model_reused'] for item in long_result['runs']] != [False, True]):
                    raise RuntimeError(f'{mode}: long job did not reuse one model for both segments')
                for segment in long_result['runs']:
                    allocated, reserved = segment['peak_allocated_bytes'], segment['peak_reserved_bytes']
                    if not 0 < allocated <= reserved <= properties.total_memory:
                        raise RuntimeError(f'{mode}: invalid long-job memory measurements')
                    if args.max_vram_gib is not None and reserved > args.max_vram_gib * 1024**3:
                        raise RuntimeError(f'{mode}: long job exceeded --max-vram-gib')
                with long_log.open('a') as log:
                    subprocess.run([*long_command, '--resume'], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=args.timeout, check=True)
                resumed = json.loads(long_metrics.read_text())
                if resumed['model_loads'] != 0 or resumed['segments_processed'] != 0 or resumed['segments_reused'] != 2:
                    raise RuntimeError(f'{mode}: completed resume repeated inference')
                result['long'] = {'metrics': long_result, 'resumed': resumed, 'log': str(long_log)}
            result['status'] = 'passed'
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
    logging.info('All three modes passed: %s', report_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(run_cli(_main))
