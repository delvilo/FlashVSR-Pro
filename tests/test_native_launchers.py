"""Shared-core orchestration with real FFmpeg splitting, audio and publication."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from flashvsr.cli import main, parse_args
from flashvsr.config import InferenceConfig
from flashvsr.media import MediaTools
from flashvsr.workflows import run_batch, run_long

PROJECT = Path(__file__).resolve().parents[1]


class NativeLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="flashvsr work '")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_all_commands_share_every_inference_option_and_validation(self):
        options = ['--mode', 'full', '--scale', '3', '--seed', '42', '--tile-dit', '--tile-vae',
                   '--tile-size', '512', '--overlap', '32', '--keep-audio', '--fps', '24',
                   '--device', 'cuda:1', '--dtype', 'fp16', '--color-fix', '--quality', '8',
                   '--sparse-ratio', '1', '--kv-ratio', '4', '--local-range', '9', '--model-dir', '/models']
        commands = [['infer', '-i', 'in.mp4'], ['batch'], ['long', '-i', 'in.mp4', '-o', 'out']]
        configs = [InferenceConfig.from_namespace(parse_args([*cmd, *options])) for cmd in commands]
        self.assertEqual(configs[0], configs[1])
        self.assertEqual(configs[0], configs[2])
        with self.assertRaises(ValueError):
            replace(configs[0], scale=float('nan'))

    def test_legacy_entry_help_works_from_another_directory(self):
        for name in ('infer.py', 'batch_inference.py', 'long_video_worker.py'):
            result = subprocess.run([sys.executable, str(PROJECT / name), '--help'], cwd=self.root,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--sparse-ratio', result.stdout)
            self.assertIn('--model-dir', result.stdout)

    def test_batch_common_engine_continues_failure_and_avoids_name_collisions(self):
        source, destination = self.root / 'in', self.root / 'out'
        source.mkdir()
        for name in ('clip.mp4', 'clip.mov', 'bad.mp4'):
            (source / name).touch()
        engine = MagicMock()
        engine.run.side_effect = [RuntimeError('decode failed'), {'status': 'ok'}, {'status': 'ok'}]
        config = InferenceConfig(seed=77, keep_audio=True, tile_dit=True)
        with patch('flashvsr.workflows.InferenceEngine', return_value=engine) as factory:
            self.assertEqual(run_batch(config, source, destination), 1)
        factory.assert_called_once_with(config)
        self.assertEqual(engine.run.call_count, 3)
        targets = [call.args[1] for call in engine.run.call_args_list]
        self.assertEqual(len(set(targets)), 3)
        report = json.loads((destination / 'batch.json').read_text())
        self.assertEqual([run['status'] for run in report['runs']], ['failed', 'ok', 'ok'])

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_long_worker_real_split_merge_audio_and_failed_output_preservation(self):
        media = MediaTools()
        source = self.root / 'source video.mp4'
        media.run(['-f', 'lavfi', '-i', 'testsrc2=size=128x96:rate=8:duration=2',
                   '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2',
                   '-f', 'lavfi', '-i', 'sine=frequency=880:duration=2',
                   '-map', '0:v', '-map', '1:a', '-map', '2:a',
                   '-c:v', 'libx264', '-g', '8', '-pix_fmt', 'yuv420p', '-c:a', 'aac', source])
        config = InferenceConfig(scale=1, keep_audio=True, seed=9, dtype='fp16')
        engine = MagicMock()
        def copy_frames(part, target, **kwargs):
            media.save_video(kwargs['input_frames'], target, fps=8, lossless=True)
            return {'status': 'ok', 'peak_allocated_bytes': 1}
        engine.model_loads = 1
        engine.registry.identity.return_value = {'version': 'test', 'revision': 'test'}
        engine.run.side_effect = copy_frames
        target_dir = self.root / 'results'
        with patch('flashvsr.long_video.InferenceEngine', return_value=engine) as factory:
            self.assertEqual(run_long(config, source, target_dir, segment_time=1), 0)
        self.assertEqual(factory.call_args.args[0], replace(config, keep_audio=False))
        self.assertEqual(engine.run.call_count, 2)
        final = target_dir / 'FlashVSR_source video_Final.mp4'
        media.verify_video(final, width=128, height=96, frames=16, fps=8, audio_streams=2)
        previous = final.read_bytes()
        engine.run.side_effect = RuntimeError('inference failed')
        failed_work = self.root / 'failed-job'
        with patch('flashvsr.long_video.InferenceEngine', return_value=engine):
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                run_long(config, source, target_dir, segment_time=1, work_dir=failed_work)
        self.assertTrue((failed_work / 'job.json').is_file())
        self.assertEqual(final.read_bytes(), previous)
        self.assertEqual(json.loads(Path(f'{final}.json').read_text())['status'], 'failed')

    def test_usage_errors_precede_creation(self):
        with self.assertRaises(SystemExit) as error:
            main(['long', '-i', 'missing.mp4', '-o', str(self.root / 'out'), '--segment-time', '0'])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_log_cannot_share_future_batch_output_or_report(self):
        from flashvsr.config import output_path
        source, destination = self.root / 'in', self.root / 'out.v1'
        source.mkdir()
        video = source / 'clip.mp4'
        video.write_bytes(b'input')
        target = output_path(video, destination, InferenceConfig(), directory=True)
        for log in (target, Path(f'{target}.json'), destination / 'batch.json'):
            with patch('flashvsr.cli.configure_logging') as configure:
                code = main(['batch', '--input-dir', str(source), '--output-dir', str(destination), '--log-file', str(log)])
            self.assertEqual(code, 1)
            configure.assert_not_called()
            self.assertFalse(log.exists())
