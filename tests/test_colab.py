"""Drive restarts and cache integrity without requiring Google auth or a GPU."""

from dataclasses import replace
import ast
import errno
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from flashvsr.colab.cache import WheelCache, cache_key, ensure_wheel
from flashvsr.colab.setup import validate_baseline
from flashvsr.colab.storage import DriveStore, copy_verified, local_lock, prepare_models, sync_file
from flashvsr.colab.workflow import list_jobs, run_job, validate_gpu
from flashvsr.config import InferenceConfig
from flashvsr.jobs import file_identity
from flashvsr.media import MediaTools
from flashvsr.models import ModelFile, ModelRegistry


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = DriveStore(self.root / 'drive', reserve_gb=0)
        self.source = self.root / 'source'
        self.source.write_bytes(b'verified data')

    def test_corrupt_copy_does_not_replace_previous_result(self):
        destination = self.root / 'output'
        destination.write_bytes(b'previous result')
        bad = {'bytes': 13, 'sha256': 'wrong'}
        with self.assertRaisesRegex(ValueError, 'verification failed'):
            copy_verified(self.source, destination, bad)
        self.assertEqual(destination.read_bytes(), b'previous result')
        self.assertFalse(list(self.root.glob('.flashvsr-*.part')))

    def test_budget_counts_temporary_copy_before_overwriting(self):
        self.store.copy(self.source, 'result')
        self.store.budget = 20
        self.source.write_bytes(b'new data of the same length')
        with self.assertRaisesRegex(OSError, 'budget'):
            self.store.copy(self.source, 'result')
        self.assertEqual(self.store.path('result').read_bytes(), b'verified data')

    def test_failed_write_preserves_previous_and_cleans_partial(self):
        self.store.copy(self.source, 'result')
        self.source.write_bytes(b'new result')
        with patch('flashvsr.colab.storage.shutil.copyfileobj', side_effect=OSError('Drive I/O error')):
            with self.assertRaisesRegex(OSError, 'Drive I/O'):
                self.store.copy(self.source, 'result')
        self.assertEqual(self.store.path('result').read_bytes(), b'verified data')
        self.assertFalse(list(self.store.root.glob('.flashvsr-*.part')))

    def test_paths_reject_traversal_and_symlinks(self):
        for path in ('../outside', '/absolute'):
            with self.assertRaises(ValueError):
                self.store.copy(self.source, path)
        (self.store.root / 'escape').symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            self.store.copy(self.source, 'escape/source')

    def test_only_owned_partial_files_are_recovered(self):
        (self.store.root / '.flashvsr-interrupted.part').write_text('partial')
        (self.store.root / 'user.part').write_text('keep')
        self.store.recover_partials()
        self.assertFalse((self.store.root / '.flashvsr-interrupted.part').exists())
        self.assertEqual((self.store.root / 'user.part').read_text(), 'keep')

    def test_fuse_fsync_limitations_do_not_hide_real_io_errors(self):
        with self.source.open('a') as stream:
            with patch('os.fsync', side_effect=OSError(errno.ENOTSUP, 'unsupported')):
                sync_file(stream)
            with patch('os.fsync', side_effect=OSError(errno.EIO, 'disk failure')):
                with self.assertRaises(OSError):
                    sync_file(stream)

    def test_runtime_lock_is_exclusive_and_releases_on_failure(self):
        lock = self.root / 'runtime.lock'
        with self.assertRaisesRegex(RuntimeError, 'active'):
            with local_lock(lock), local_lock(lock):
                pass
        with local_lock(lock):
            pass

    def test_invalid_budget_is_rejected(self):
        for budget, reserve in ((20, 20), (0, 0), (-1, 0), (float('nan'), 0)):
            with self.assertRaises(ValueError):
                DriveStore(self.root / 'invalid', budget, reserve)

    def test_model_cache_downloads_selected_mode_and_restores_after_vm_reset(self):
        payloads = {'tiny.bin': b'tiny weight', 'full.bin': b'full weight'}
        entries = tuple(ModelFile(name, len(data), hashlib.sha256(data).hexdigest(),
                                  ('tiny',) if name.startswith('tiny') else ('full',))
                        for name, data in payloads.items())
        downloaded = []
        def registry(*args, **kwargs):
            instance = ModelRegistry(*args, **kwargs)
            instance.entries = entries
            return instance
        def download(instance, mode):
            for item in instance.selected(mode):
                downloaded.append(item.name)
                instance.directory.mkdir(parents=True, exist_ok=True)
                (instance.directory / item.name).write_bytes(payloads[item.name])
        with patch('flashvsr.models.ModelRegistry', side_effect=registry), patch.object(ModelRegistry, 'download', download):
            prepare_models(self.store, self.root / 'local', 'tiny')
            shutil.rmtree(self.root / 'local')
            prepare_models(self.store, self.root / 'local', 'tiny')
            self.assertEqual(downloaded, ['tiny.bin'])
            self.assertFalse(self.store.path('models/v1.1/full.bin').exists())
            self.store.path('models/v1.1/tiny.bin').write_bytes(b'broken file')
            shutil.rmtree(self.root / 'local')
            prepare_models(self.store, self.root / 'local', 'tiny')
            self.assertEqual(downloaded, ['tiny.bin', 'tiny.bin'])


class WheelCacheTests(unittest.TestCase):
    # Keep the cache tests on real files; callbacks stand in for pip and CUDA.
    setUp = StorageTests.setUp
    def cache(self):
        return WheelCache(self.store, {'python_abi': 'cp313', 'torch': '2.11.0+cu128',
                                     'cuda': '12.8', 'gpu': [8, 0], 'source': 'revision-a'})

    def test_incompatible_identities_never_reuse_a_wheel(self):
        identity = self.cache().identity
        for key in identity:
            with self.subTest(key=key):
                self.assertNotEqual(cache_key(identity), cache_key({**identity, key: 'changed'}))

    def test_cache_hit_skips_build_and_still_checks_gpu(self):
        cache = self.cache()
        wheel = self.root / 'block_sparse_attn-test.whl'
        wheel.write_bytes(b'wheel payload')
        cache.publish(wheel)
        build, install, smoke = MagicMock(), MagicMock(), MagicMock()
        result = ensure_wheel(cache, self.root / 'build', build, install, smoke)
        self.assertEqual(result['status'], 'reused')
        build.assert_not_called()
        install.assert_called_once()
        smoke.assert_called_once()

    def test_corrupt_wheel_is_rebuilt_and_failed_gpu_test_is_not_published(self):
        cache = self.cache()
        wheel = self.root / 'block_sparse_attn-test.whl'
        wheel.write_bytes(b'original')
        cache.publish(wheel)
        cached = self.store.path(cache.directory / wheel.name)
        cached.write_bytes(b'corrupt!')
        wheel.write_bytes(b'new wheel')
        build = MagicMock(return_value=wheel)
        with self.assertRaisesRegex(RuntimeError, 'kernel failed'):
            ensure_wheel(cache, self.root / 'build', build, MagicMock(), MagicMock(side_effect=RuntimeError('kernel failed')))
        self.assertEqual(cached.read_bytes(), b'corrupt!')
        result = ensure_wheel(cache, self.root / 'build', build, MagicMock(), MagicMock())
        self.assertEqual(result['status'], 'built')
        self.assertEqual(cached.read_bytes(), b'new wheel')

    def test_failed_cached_gpu_test_rebuilds_once(self):
        cache = self.cache()
        wheel = self.root / 'block_sparse_attn-test.whl'
        wheel.write_bytes(b'wheel')
        cache.publish(wheel)
        build = MagicMock(return_value=wheel)
        smoke = MagicMock(side_effect=[RuntimeError('incompatible'), None])
        result = ensure_wheel(cache, self.root / 'build', build, MagicMock(), smoke)
        self.assertEqual(result['status'], 'built')
        build.assert_called_once()
        self.assertEqual(smoke.call_count, 2)


class ColabBaselineTests(unittest.TestCase):
    def test_notebook_cells_are_valid_python_and_cli_help_needs_no_gpu(self):
        from flashvsr.colab.__main__ import parser
        project = Path(__file__).resolve().parents[1]
        notebook = json.loads((project / 'colab/FlashVSR_Pro.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
        with self.assertRaises(SystemExit) as result:
            parser().parse_args(['--help'])
        self.assertEqual(result.exception.code, 0)

    def test_only_current_colab_a100_l4_are_accepted(self):
        torch = SimpleNamespace(__version__='2.11.0+cu128', version=SimpleNamespace(cuda='12.8'), cuda=MagicMock())
        torch.cuda.is_available.return_value = True
        torch.cuda.current_device.return_value = 0
        torch.cuda.device_count.return_value = 1
        torch.cuda.is_bf16_supported.return_value = True
        noble = {'ID': 'ubuntu', 'VERSION_ID': '24.04', 'VERSION_CODENAME': 'noble'}
        for name, capability in (('NVIDIA A100-SXM4-40GB', (8, 0)), ('NVIDIA L4', (8, 9))):
            torch.cuda.get_device_name.return_value = name
            torch.cuda.get_device_capability.return_value = capability
            self.assertEqual(validate_baseline(torch, noble, (3, 13)), 0)
        for version in ((3, 12), (3, 14)):
            with self.assertRaisesRegex(RuntimeError, 'native Python 3.13'):
                validate_baseline(torch, noble, version)
        with self.assertRaisesRegex(RuntimeError, '24.04'):
            validate_baseline(torch, {**noble, 'VERSION_ID': '22.04'}, (3, 13))
        torch.cuda.get_device_capability.return_value = (7, 5)
        with self.assertRaisesRegex(RuntimeError, 'T4/P100'):
            validate_baseline(torch, noble, (3, 13))


class GPUValidationStorageTests(unittest.TestCase):
    setUp = StorageTests.setUp

    def test_failed_gpu_validation_saves_evidence_before_propagating_failure(self):
        def failed(command, **kwargs):
            directory = Path(command[-1])
            (directory / 'report.json').write_text('{"status":"failed"}')
            kwargs['stdout'].write('GPU failure details')
            raise subprocess.CalledProcessError(1, command)
        with patch('flashvsr.colab.workflow.subprocess.run', side_effect=failed):
            with self.assertRaises(subprocess.CalledProcessError):
                validate_gpu(self.root, self.store, self.root / 'local')
        saved, = self.store.path('diagnostics/gpu-validation').glob('*/saved.json')
        self.assertEqual(json.loads(saved.read_text())['status'], 'failed')
        self.assertEqual((saved.parent / 'runner.log').read_text(), 'GPU failure details')

    def test_validation_results_use_same_budget_guard(self):
        self.store.budget = 1
        def passed(command, **kwargs):
            kwargs['stdout'].write('GPU success details')
        with patch('flashvsr.colab.workflow.subprocess.run', side_effect=passed):
            with self.assertRaisesRegex(OSError, 'budget'):
                validate_gpu(self.root, self.store, self.root / 'local')
        self.assertFalse(list(self.store.root.rglob('saved.json')))
        self.assertTrue(list((self.root / 'local').rglob('runner.log')))


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class ColabWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.local = self.root / 'local'
        self.store = DriveStore(self.root / 'drive', reserve_gb=0)
        self.source = self.root / 'input.mp4'
        self.media = MediaTools()
        self.media.run(['-f', 'lavfi', '-i', 'testsrc2=size=32x24:rate=8:duration=2.125',
                        '-c:v', 'libx264', '-g', '999', self.source])
        self.config = InferenceConfig(scale=1)
        self.addCleanup(patch.stopall)
        patch('flashvsr.colab.workflow.project_identity', return_value='pinned-commit').start()
        patch('flashvsr.colab.workflow.prepare_models').start()

    def run_job(self, fail_at=None, resume=False):
        from test_long_video import CopyEngine
        engine = CopyEngine(self.root, self.media, fail_at=fail_at)
        with patch('flashvsr.long_video.InferenceEngine', return_value=engine):
            result = run_job(self.root, self.store, self.local, 'test-job', self.config,
                             None if resume else self.source, resume=resume, segment_frames=5)
        return result, engine

    def saved(self):
        return json.loads(self.store.path('jobs/test-job/checkpoint/job.json').read_text())

    def test_vm_reset_restores_segments_and_preserves_final_video_and_reports(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_job(fail_at=5)
        self.assertEqual(self.saved()['segments'][0]['status'], 'done')
        shutil.rmtree(self.local)
        result, engine = self.run_job(resume=True)
        self.assertEqual(engine.starts, [5, 10, 15])
        self.assertEqual(result['status'], 'ok')
        self.media.verify_video(self.store.path(result['video']), frames=17, width=32, height=24)
        for path, identity in result['files'].items():
            self.assertEqual(file_identity(self.store.path(path)), identity)
        self.assertFalse(list(self.store.path('jobs/test-job/checkpoint/segments').glob('*.mkv')))
        shutil.rmtree(self.local)
        _, completed_engine = self.run_job(resume=True)
        self.assertEqual(completed_engine.starts, [])
        self.assertTrue(list_jobs(self.store)[0]['completed'])

    def test_failed_segment_upload_does_not_advance_drive_record(self):
        original = self.store.copy
        def upload(source, relative, identity=None):
            if str(relative).endswith('000001.mkv'):
                raise OSError('Drive quota exceeded')
            return original(source, relative, identity)
        with patch.object(self.store, 'copy', side_effect=upload):
            with self.assertRaisesRegex(OSError, 'quota'):
                self.run_job()
        self.assertEqual([i['status'] for i in self.saved()['segments']], ['done', 'pending', 'pending', 'pending'])
        # The unsynced local segment is still usable in the same VM.
        _, engine = self.run_job(resume=True)
        self.assertEqual(engine.starts, [10, 15])

    def test_registration_retry_reuses_source_after_manifest_write_failure(self):
        original = self.store.write_json
        def write(relative, data):
            if str(relative).endswith('run.json'):
                raise OSError('Drive registration failed')
            return original(relative, data)
        with patch.object(self.store, 'write_json', side_effect=write):
            with self.assertRaisesRegex(OSError, 'registration failed'):
                self.run_job()
        self.assertTrue(self.store.path('jobs/test-job/source.mp4').is_file())
        result, _ = self.run_job()
        self.assertEqual(result['status'], 'ok')

    def test_failed_final_upload_keeps_segments_and_retry_needs_no_model(self):
        original = self.store.copy
        def upload(source, relative, identity=None):
            if str(relative).endswith('enhanced.mp4'):
                raise OSError('Drive result upload failed')
            return original(source, relative, identity)
        with patch.object(self.store, 'copy', side_effect=upload):
            with self.assertRaisesRegex(OSError, 'upload failed'):
                self.run_job()
        self.assertEqual(len(list(self.store.path('jobs/test-job/checkpoint/segments').glob('*.mkv'))), 4)
        self.assertFalse(self.store.path('jobs/test-job/completion.json').exists())
        _, engine = self.run_job(resume=True)
        self.assertEqual(engine.model_loads, 0)

    def test_corrupt_drive_segment_is_recomputed_after_reset(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_job(fail_at=10)
        cached = self.store.path('jobs/test-job/checkpoint/segments/000000.mkv')
        data = bytearray(cached.read_bytes())
        data[-1] ^= 1
        cached.write_bytes(data)
        shutil.rmtree(self.local)
        _, engine = self.run_job(resume=True)
        self.assertEqual(engine.starts, [0, 10, 15])

    def test_resume_refuses_changed_parameters_or_code_and_fresh_job_collision(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_job(fail_at=5)
        self.config = replace(self.config, seed=17)
        with self.assertRaisesRegex(ValueError, 'settings/code/runtime changed'):
            self.run_job(resume=True)
        self.config = replace(self.config, seed=0)
        with patch('flashvsr.colab.workflow.project_identity', return_value='new-commit'):
            with self.assertRaisesRegex(ValueError, 'settings/code/runtime changed'):
                self.run_job(resume=True)
        with self.assertRaisesRegex(FileExistsError, 'already exists'):
            self.run_job()
        # Removing job metadata must not make an old result eligible for overwrite.
        shutil.rmtree(self.store.path('jobs/test-job'))
        shutil.rmtree(self.local)
        existing = self.store.path('results/test-job/enhanced.mp4')
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b'previous result')
        with self.assertRaisesRegex(FileExistsError, 'already exists'):
            self.run_job()
        self.assertEqual(existing.read_bytes(), b'previous result')


if __name__ == '__main__':
    unittest.main()
