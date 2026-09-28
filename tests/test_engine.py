"""CPU tests of orchestration and diagnostics; these do not validate GPU math."""
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

from flashvsr.config import InferenceConfig
from flashvsr.engine import InferenceEngine, pipeline_options
from flashvsr.observability import JsonFormatter, run_id


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'input.mp4'
        self.source.write_bytes(b'input')
        self.destination = self.root / 'out.mp4'
        self.config = InferenceConfig(seed=17, keep_audio=True)
        self.registry = MagicMock()
        self.registry.identity.return_value = {'version': 'v1.1', 'revision': 'pinned', 'files': {}}
        self.media = MagicMock(ffmpeg='/usr/bin/ffmpeg', ffprobe='/usr/bin/ffprobe')
        self.media.verify_video.return_value = {'frames': 2, 'fps': 8, 'width': 4, 'height': 4}
        self.media.save_video.return_value = {'frames': 2, 'fps': 8, 'width': 8, 'height': 8, 'audio_streams': 1}
        self.engine = InferenceEngine(self.config, self.media, self.registry)
        self.pipe = MagicMock(return_value=torch.zeros(1, 3, 9, 128, 128))
        self.closed = False
        self.loads = 0
        @contextmanager
        def pipeline(*args):
            self.loads += 1
            try:
                yield self.pipe
            finally:
                self.closed = True
        patches = [patch('utils.runtime.validate_cuda', return_value=0),
                   patch('flashvsr.model_loading.load_pipeline', pipeline),
                   patch('flashvsr.frames.prepare_input_tensor', return_value=(
                       torch.zeros(1, 3, 9, 128, 128), 128, 128, 9, 8, str(self.source), 2, 8, 8)),
                   patch('torch.cuda.get_device_name', return_value='CPU test double'),
                   patch('torch.cuda.current_device', return_value=0),
                   patch('torch.cuda.reset_peak_memory_stats'), patch('torch.cuda.synchronize'),
                   patch('torch.cuda.empty_cache'),
                   patch('torch.cuda.max_memory_allocated', return_value=100),
                   patch('torch.cuda.max_memory_reserved', return_value=200)]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def report(self):
        return json.loads(Path(f'{self.destination}.json').read_text())

    def test_success_records_parameters_version_stages_fps_and_memory(self):
        result = self.engine.run(self.source, self.destination)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['parameters']['seed'], 17)
        self.assertEqual(result['models']['revision'], 'pinned')
        self.assertEqual(result['peak_allocated_bytes'], 100)
        self.assertEqual(result['peak_reserved_bytes'], 200)
        self.assertGreater(result['inference_fps'], 0)
        self.assertGreater(result['end_to_end_fps'], 0)
        self.assertEqual(set(result['timings']), {'preflight', 'model_load', 'decode', 'inference', 'postprocess', 'encode'})
        self.assertEqual(self.pipe.call_args.kwargs['seed'], 17)
        self.assertEqual(len(self.media.save_video.call_args.args[0]), 2)
        self.assertEqual(self.media.save_video.call_args.kwargs['audio_source'], str(self.source))
        self.assertTrue(self.closed)
        self.assertEqual(self.report()['run_id'], result['run_id'])

    def test_failure_records_phase_and_always_cleans_pipeline(self):
        self.media.save_video.side_effect = RuntimeError('encoder failed')
        with self.assertRaisesRegex(RuntimeError, 'encoder failed'):
            self.engine.run(self.source, self.destination)
        report = self.report()
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['stage'], 'encode')
        self.assertEqual(report['error']['message'], 'encoder failed')
        self.assertEqual(report['peak_reserved_bytes'], 200)
        self.assertTrue(self.closed)

    def test_corrupt_model_stops_before_any_cuda_or_media_calls(self):
        self.registry.check.side_effect = ValueError('SHA-256 mismatch')
        with patch('utils.runtime.validate_cuda') as cuda:
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                self.engine.run(self.source, self.destination)
        cuda.assert_not_called()
        self.media.verify_video.assert_not_called()
        self.assertEqual(self.report()['stage'], 'preflight')

    def test_model_load_failure_is_attributed_to_model_load(self):
        with patch('flashvsr.model_loading.load_pipeline', side_effect=RuntimeError('bad state dict')):
            with self.assertRaisesRegex(RuntimeError, 'bad state dict'):
                self.engine.run(self.source, self.destination)
        self.assertEqual(self.report()['stage'], 'model_load')

    def test_report_collision_cannot_overwrite_input_or_output(self):
        for path in (self.source, self.destination):
            with self.assertRaisesRegex(ValueError, 'different path'):
                self.engine.run(self.source, self.destination, path)
        self.assertEqual(self.source.read_bytes(), b'input')
        self.registry.check.assert_not_called()

    def test_report_cannot_replace_model_weights(self):
        from flashvsr.models import ModelFile
        weight = self.root / 'weight.ckpt'
        weight.write_bytes(b'weights')
        self.registry.directory = self.root
        self.registry.entries = (ModelFile(weight.name, 7, 'digest', ('tiny',)),)
        with self.assertRaisesRegex(ValueError, 'different path'):
            self.engine.run(self.source, self.destination, weight)
        self.assertEqual(weight.read_bytes(), b'weights')

    def test_tiling_uses_pixel_to_latent_conversion_and_scaled_sparse_ratio(self):
        options = pipeline_options(InferenceConfig(mode='full', tile_vae=True, tile_size=512, overlap=64), None, 256, 384, 17)
        self.assertEqual(options['tile_size'], (64, 64))
        self.assertEqual(options['tile_stride'], (56, 56))
        self.assertEqual(options['topk_ratio'], 20)

    def test_structured_log_records_run_id_and_level(self):
        token = run_id.set('test-run')
        try:
            record = logging.LogRecord('flashvsr.engine', logging.WARNING, '', 1, '壞檔案 %s', ('clip.mp4',), None)
            data = json.loads(JsonFormatter().format(record))
        finally:
            run_id.reset(token)
        self.assertEqual(data['run_id'], 'test-run')
        self.assertEqual(data['level'], 'WARNING')
        self.assertEqual(data['message'], '壞檔案 clip.mp4')

    def test_session_reuses_weights_and_clears_video_state_between_runs(self):
        with self.engine.session():
            first = self.engine.run(self.source, self.destination)
            second = self.engine.run(self.source, self.destination)
            self.assertFalse(self.closed)
            self.assertFalse(first['model_reused'])
            self.assertTrue(second['model_reused'])
        self.assertEqual(self.loads, 1)
        self.registry.check.assert_called_once()
        self.assertTrue(self.closed)
        self.assertEqual(self.pipe.dit.LQ_proj_in.clear_cache.call_count, 4)
        self.assertEqual(self.pipe.TCDecoder.clean_mem.call_count, 4)
        self.pipe.dit.clear_cross_kv.assert_not_called()

    def test_failed_session_run_discards_model_before_retry(self):
        with self.engine.session():
            self.media.save_video.side_effect = RuntimeError('encode failed')
            with self.assertRaisesRegex(RuntimeError, 'encode failed'):
                self.engine.run(self.source, self.destination)
            self.assertIsNone(self.engine._pipe)
            self.media.save_video.side_effect = None
            result = self.engine.run(self.source, self.destination)
        self.assertFalse(result['model_reused'])
        self.assertEqual(self.loads, 2)

    def test_session_closes_pipeline_on_interruption(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.engine.session():
                self.engine.run(self.source, self.destination)
                raise KeyboardInterrupt
        self.assertTrue(self.closed)
        self.assertIsNone(self.engine._pipe)
