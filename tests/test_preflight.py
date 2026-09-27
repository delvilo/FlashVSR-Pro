"""CLI and environment failures must happen before model/GPU initialization."""

from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from utils.cli import parse_args, run_cli, validate_input, validate_output
from utils.runtime import require_weight, validate_cuda, validate_models, validate_python, validate_toolkit

PROJECT = Path(__file__).resolve().parents[1]


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_python_requires_supported_cuda_wheels(self):
        for version in ((3, 10), (3, 11), (3, 15)):
            with self.subTest(version=version), patch('utils.runtime.sys.version_info', version):
                with self.assertRaisesRegex(RuntimeError, 'Python 3.12–3.14'):
                    validate_python()
        for version in ((3, 12), (3, 13), (3, 14)):
            with self.subTest(version=version), patch('utils.runtime.sys.version_info', version):
                validate_python()

    def test_usage_rejects_invalid_values(self):
        cases = [('--scale', 'nan'), ('--scale', 'inf'), ('--scale', '0'),
                 ('--fps', '-1'), ('--overlap', '128'), ('--tile-size', '129'),
                 ('--device', 'cpu'), ('--dtype', 'fp32'), ('--seed', '-1'),
                 ('--local-range', '0'), ('--quality', '11'), ('--mode', 'other')]
        for args in cases:
            with self.subTest(args=args), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_args(['-i', 'input.mp4', *args])
                self.assertEqual(error.exception.code, 2)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['-i', 'input.mp4', '--tile-vae'])

    def test_missing_empty_and_lfs_weights(self):
        path = self.root / 'weight.pth'
        with self.assertRaises(FileNotFoundError):
            require_weight(path)
        path.touch()
        with self.assertRaises(FileNotFoundError):
            require_weight(path)
        path.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:123\n')
        with self.assertRaisesRegex(ValueError, 'Git LFS pointer'):
            require_weight(path)

    def test_mode_specific_weights(self):
        models = self.root / 'models'
        models.mkdir()
        prompt = self.root / 'models/prompt_tensor/posi_prompt.pth'
        prompt.parent.mkdir()
        prompt.write_bytes(b'prompt')
        for name in ('diffusion_pytorch_model_streaming_dmd.safetensors', 'LQ_proj_in.ckpt', 'TCDecoder.ckpt'):
            (models / name).write_bytes(b'weight')
        for mode in ('tiny', 'tiny-long'):
            self.assertEqual(validate_models(mode, models, self.root), models)
        with self.assertRaisesRegex(FileNotFoundError, 'Wan2.1_VAE.pth'):
            validate_models('full', models, self.root)
        (models / 'Wan2.1_VAE.pth').write_bytes(b'weight')
        validate_models('full', models, self.root)

    def test_cuda_rejects_cpu_and_unsupported_hardware_before_set_device(self):
        torch = SimpleNamespace(__version__='2.10.0+cu126', version=SimpleNamespace(cuda='12.6'), cuda=MagicMock())
        torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
            validate_cuda(torch)
        torch.cuda.set_device.assert_not_called()
        torch.cuda.is_available.return_value = True
        torch.cuda.current_device.return_value = 0
        torch.cuda.device_count.return_value = 1
        torch.cuda.get_device_capability.return_value = (7, 5)
        with self.assertRaisesRegex(RuntimeError, 'T4/P100'):
            validate_cuda(torch)
        torch.cuda.set_device.assert_not_called()
        torch.cuda.get_device_capability.return_value = (8, 9)
        torch.cuda.is_bf16_supported.return_value = True
        self.assertEqual(validate_cuda(torch), 0)
        with self.assertRaisesRegex(ValueError, 'index'):
            validate_cuda(torch, 'cuda:1')
        with self.assertRaisesRegex(ValueError, 'CPU inference'):
            validate_cuda(torch, 'cpu')
        torch.version.cuda = '12.4'
        with self.assertRaisesRegex(RuntimeError, 'cu126'):
            validate_cuda(torch)

    def test_toolkit_must_match_baseline(self):
        nvcc = self.root / 'bin/nvcc'
        nvcc.parent.mkdir()
        nvcc.touch()
        with patch('utils.runtime.subprocess.check_output', return_value='Cuda compilation tools, release 12.4, V12.4.0'):
            with self.assertRaisesRegex(RuntimeError, 'Toolkit 12.5'):
                validate_toolkit(self.root)
        for minor in (5, 6, 8):
            with self.subTest(minor=minor), patch('utils.runtime.subprocess.check_output', return_value=f'Cuda compilation tools, release 12.{minor}, V12.{minor}.0'):
                self.assertIn(f'12.{minor}', validate_toolkit(self.root))
        with patch('utils.runtime.subprocess.check_output', return_value='Cuda compilation tools, release 13.0, V13.0.0'):
            with self.assertRaisesRegex(RuntimeError, 'Toolkit 12.5'):
                validate_toolkit(self.root)

    def test_input_and_output_validation(self):
        source = self.root / 'clip.mp4'
        source.write_bytes(b'data')
        with self.assertRaises(FileNotFoundError):
            validate_input(self.root / 'missing.mp4')
        empty = self.root / 'empty'
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, 'No PNG/JPEG'):
            validate_input(empty)
        with self.assertRaisesRegex(ValueError, 'different files'):
            validate_output(source, source)
        with self.assertRaisesRegex(ValueError, 'GIF'):
            validate_output(self.root / 'out.gif', source, True)

    def test_shared_exit_codes(self):
        def fail(argv):
            raise RuntimeError('inference failed')
        def cancel(argv):
            raise KeyboardInterrupt
        with redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(run_cli(fail), 1)
            self.assertEqual(run_cli(cancel), 130)
        self.assertIn('inference failed', errors.getvalue())
        self.assertEqual(run_cli(lambda argv: None), 0)

    def test_help_and_missing_models_do_not_import_torch(self):
        # Deliberately fail if startup tries to import the inference dependencies.
        hook = self.root / 'sitecustomize.py'
        hook.write_text('import sys\nclass Block:\n def find_spec(self, name, *args):\n  if name == "torch": raise RuntimeError("EARLY_TORCH_IMPORT")\nsys.meta_path.insert(0, Block())\n')
        env = dict(os.environ, PYTHONPATH=str(self.root), FLASHVSR_MODEL_PATH=str(self.root / 'absent'))
        result = subprocess.run([sys.executable, str(PROJECT / 'infer.py'), '--help'], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        source = self.root / 'input.mp4'
        source.write_bytes(b'fixture')
        result = subprocess.run([sys.executable, str(PROJECT / 'infer.py'), '-i', str(source)], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('Model weight missing', result.stderr)
        self.assertNotIn('EARLY_TORCH_IMPORT', result.stderr)


if __name__ == '__main__':
    unittest.main()
