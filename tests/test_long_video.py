"""Real frame decoding/encoding with CPU inference doubles and durable job recovery."""

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import weakref

import numpy as np

from flashvsr.cli import main, parse_args
from flashvsr.config import InferenceConfig
from flashvsr.jobs import JobRecord
from flashvsr.long_video import run_long
from flashvsr.media import MediaError, MediaTools
from flashvsr.streaming import read_video_frames


class CopyEngine:
    def __init__(self, root, media, fail_at=None):
        self.registry = MagicMock()
        self.registry.directory = root / 'models'
        self.registry.entries = ()
        self.registry.identity.return_value = {'version': 'test', 'revision': 'pinned'}
        self.media = media
        self.model_loads = 0
        self.fail_at = fail_at
        self.starts = []
        self.sizes = []
        self.previous_frames = None
        self.closed = False

    @contextmanager
    def session(self):
        self.model_loads += 1
        try:
            yield self
        finally:
            self.closed = True

    def run(self, source, target, *, input_frames, media_info, frame_start, lossless):
        if self.previous_frames is not None and self.previous_frames() is not None:
            raise AssertionError('Previous segment is still retained')
        self.previous_frames = weakref.ref(input_frames)
        self.starts.append(frame_start)
        self.sizes.append(len(input_frames))
        if frame_start == self.fail_at:
            raise KeyboardInterrupt
        return {**self.media.save_video(input_frames, target, fps=media_info['fps'], lossless=lossless),
                'status': 'ok', 'peak_allocated_bytes': 1}


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class LongVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="long job '")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'input.mp4'
        self.work = self.root / 'job'
        self.output = self.root / 'out'
        self.final = self.output / 'FlashVSR_input_Final.mp4'
        self.media = MediaTools()
        # One keyframe: hard frame limits must work even inside a long GOP.
        self.media.run(['-f', 'lavfi', '-i', 'testsrc2=size=32x24:rate=8:duration=2.125',
                        '-c:v', 'libx264', '-g', '999', '-keyint_min', '999', '-sc_threshold', '0', self.source])
        self.config = InferenceConfig(scale=1)

    def run_job(self, engine, **kwargs):
        with patch('flashvsr.long_video.InferenceEngine', return_value=engine):
            return run_long(self.config, self.source, self.output, segment_frames=5, work_dir=self.work, **kwargs)

    def record(self):
        return json.loads((self.work / 'job.json').read_text())

    def test_hard_frame_bound_and_lossless_cache_preserve_order(self):
        engine = CopyEngine(self.root, self.media)
        self.assertEqual(self.run_job(engine, keep_temp=True), 0)
        self.assertEqual(engine.sizes, [5, 5, 5, 2])
        self.assertEqual(engine.starts, [0, 5, 10, 15])
        self.assertTrue(engine.closed)
        original = self.media.verify_video(self.source)
        with read_video_frames(self.media, self.source, original) as reader:
            expected = reader.read(17)
        for item in self.record()['segments']:
            path = self.work / 'segments' / f"{item['index']:06d}.mkv"
            with read_video_frames(self.media, path, self.media.verify_video(path)) as reader:
                actual = reader.read(item['frames'])
            np.testing.assert_array_equal(actual, expected[item['start']:item['start']+item['frames']])
        self.media.verify_video(self.final, frames=17, width=32, height=24, fps=8, audio_streams=0)

    def test_interruption_resume_and_completed_job_need_no_repeated_inference(self):
        broken = CopyEngine(self.root, self.media, fail_at=5)
        with self.assertRaises(KeyboardInterrupt):
            self.run_job(broken)
        self.assertTrue(broken.closed)
        self.assertEqual(self.record()['status'], 'cancelled')
        self.assertEqual(self.record()['segments'][0]['status'], 'done')
        resumed = CopyEngine(self.root, self.media)
        self.assertEqual(self.run_job(resumed, resume=True), 0)
        self.assertEqual(resumed.starts, [5, 10, 15])
        self.assertEqual(list((self.work / 'segments').glob('*.mkv')), [])
        finished = CopyEngine(self.root, self.media)
        self.assertEqual(self.run_job(finished, resume=True), 0)
        self.assertEqual(finished.model_loads, 0)
        self.assertEqual(finished.starts, [])

    def test_corrupt_cached_segment_is_reprocessed_but_other_valid_segment_is_reused(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_job(CopyEngine(self.root, self.media, fail_at=10))
        first = self.work / 'segments/000000.mkv'
        data = bytearray(first.read_bytes())
        data[-1] ^= 1
        first.write_bytes(data)
        resumed = CopyEngine(self.root, self.media)
        self.assertEqual(self.run_job(resumed, resume=True), 0)
        self.assertEqual(resumed.starts, [0, 10, 15])

    def test_encoding_failure_preserves_old_output_and_resume_skips_all_models(self):
        self.output.mkdir()
        self.final.write_bytes(b'previous output')
        with patch('flashvsr.media.MediaTools.encode_segments', side_effect=MediaError('final encode failed')):
            with self.assertRaisesRegex(MediaError, 'final encode failed'):
                self.run_job(CopyEngine(self.root, self.media))
        self.assertEqual(self.final.read_bytes(), b'previous output')
        self.assertTrue(all(item['status'] == 'done' for item in self.record()['segments']))
        resumed = CopyEngine(self.root, self.media)
        self.assertEqual(self.run_job(resumed, resume=True), 0)
        self.assertEqual(resumed.model_loads, 0)
        self.assertEqual(resumed.starts, [])

    def test_resume_refuses_changed_settings_and_new_run_refuses_existing_job(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_job(CopyEngine(self.root, self.media, fail_at=5))
        self.config = replace(self.config, seed=17)
        with self.assertRaisesRegex(ValueError, 'Cannot resume'):
            self.run_job(CopyEngine(self.root, self.media), resume=True)
        self.config = replace(self.config, seed=0)
        with self.assertRaisesRegex(FileExistsError, '--resume'):
            self.run_job(CopyEngine(self.root, self.media))

    def test_frame_reader_detects_truncation_and_closes_decoder_on_cancel(self):
        original = self.media.verify_video(self.source)
        with self.assertRaisesRegex(MediaError, 'expected frame count'):
            with read_video_frames(self.media, self.source, original) as reader:
                reader.read(18)
        self.assertIsNotNone(reader.process.poll())
        with self.assertRaises(KeyboardInterrupt):
            with read_video_frames(self.media, self.source, original) as reader:
                reader.read(1)
                raise KeyboardInterrupt
        self.assertIsNotNone(reader.process.poll())

    def test_final_nvenc_failure_retries_software_from_lossless_segments(self):
        original_run = MediaTools.run
        attempts = []
        def encode(media, arguments, **kwargs):
            if '-c:v' in arguments:
                codec = arguments[arguments.index('-c:v')+1]
                attempts.append(codec)
                if codec == 'hevc_nvenc':
                    raise MediaError('GPU encode failed')
            return original_run(media, arguments, **kwargs)
        with patch.object(MediaTools, 'nvenc_works', return_value=True), patch.object(MediaTools, 'run', encode):
            self.assertEqual(self.run_job(CopyEngine(self.root, self.media)), 0)
        self.assertEqual(attempts, ['hevc_nvenc', 'libx264'])
        self.media.verify_video(self.final, frames=17, video_codec='h264')


class JobRecordTests(unittest.TestCase):
    def test_resume_rejects_changed_source_models_or_runtime_without_replacing_record(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = {'input_media': {'frames': 7}, 'segment_frames': 3, 'input_identity': 'original',
                    'models': 'pinned', 'runtime': 'baseline'}
            with JobRecord(Path(temp), spec, False).open():
                pass
            record = (Path(temp) / 'job.json').read_bytes()
            for field in ('input_identity', 'models', 'runtime'):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Cannot resume'):
                    with JobRecord(Path(temp), {**spec, field: 'changed'}, True).open():
                        pass
            self.assertEqual((Path(temp) / 'job.json').read_bytes(), record)

    def test_job_cache_and_reports_cannot_follow_symlinks_outside_job(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outside = root / 'important.json'
            outside.write_text('keep me')
            with JobRecord(root / 'job', {'input_media': {'frames': 1}, 'segment_frames': 1}, False).open() as job:
                (job.directory / 'segments/000000.mkv.json').symlink_to(outside)
                with self.assertRaisesRegex(ValueError, 'escapes'):
                    job.segment_path(0)
            self.assertEqual(outside.read_text(), 'keep me')

    def test_lock_is_exclusive_and_released_without_deleting_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            spec = {'input_media': {'frames': 7}, 'segment_frames': 3}
            with JobRecord(Path(temp), spec, False).open() as first:
                with self.assertRaisesRegex(RuntimeError, 'already running'):
                    with JobRecord(Path(temp), spec, True).open():
                        pass
                self.assertEqual([item['frames'] for item in first.data['segments']], [3, 3, 1])
            with JobRecord(Path(temp), spec, True).open() as resumed:
                self.assertEqual(len(resumed.data['segments']), 3)

    def test_cli_checks_frame_limit_and_prevents_log_overwriting_checkpoint(self):
        with self.assertRaises(SystemExit) as error:
            parse_args(['long', '-i', 'in.mp4', '-o', 'out', '--segment-frames', '0'])
        self.assertEqual(error.exception.code, 2)
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp) / 'job'
            with patch('flashvsr.cli.configure_logging') as configure:
                code = main(['long', '-i', 'in.mp4', '-o', 'out', '--work-dir', str(work),
                             '--log-file', str(work / 'job.json')])
            self.assertEqual(code, 1)
            configure.assert_not_called()
            self.assertFalse(work.exists())
