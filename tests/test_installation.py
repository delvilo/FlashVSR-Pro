"""Installer failure ordering and dependency metadata contracts."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

try:
    import tomllib
except ImportError:
    import tomli as tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

PROJECT = Path(__file__).resolve().parents[1]


class InstallationTests(unittest.TestCase):
    def test_runtime_has_one_opencv_and_no_build_tools(self):
        requirements = [Requirement(line) for line in (PROJECT / 'requirements.txt').read_text().splitlines()
                        if line.strip() and not line.startswith('#')]
        names = [canonicalize_name(item.name) for item in requirements]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([name for name in names if name.startswith('opencv-')], ['opencv-python-headless'])
        self.assertTrue(all(str(item.specifier).startswith('==') for item in requirements))
        self.assertFalse(set(names) & {'pip', 'setuptools', 'wheel', 'ninja', 'build'})
        metadata = tomllib.loads((PROJECT / 'pyproject.toml').read_text())
        self.assertEqual(metadata['tool']['setuptools']['dynamic']['dependencies']['file'], ['requirements.txt'])
        self.assertEqual(metadata['project']['requires-python'], '>=3.10,<3.13')

    def test_installer_help_and_usage(self):
        help_result = subprocess.run(['bash', str(PROJECT / 'scripts/install.sh'), '--help'], capture_output=True)
        self.assertEqual(help_result.returncode, 0)
        invalid = subprocess.run(['bash', str(PROJECT / 'scripts/install.sh'), '--unknown'], capture_output=True)
        self.assertEqual(invalid.returncode, 2)

    def test_incompatible_cuda_stops_before_pip_changes_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bin_dir = root / 'bin'
            bin_dir.mkdir()
            nvcc = bin_dir / 'nvcc'
            nvcc.write_text('#!/bin/sh\necho "Cuda compilation tools, release 12.6, V12.6.0"\n')
            nvcc.chmod(0o755)
            log = root / 'pip-was-called'
            wrapper = root / 'selected-python'
            wrapper.write_text('#!/bin/sh\nif [ "$1" = "-m" ] && [ "$2" = "pip" ]; then\n  touch "$FLASHVSR_TEST_PIP_LOG"\n  exit 99\nfi\nexec "$FLASHVSR_TEST_REAL_PYTHON" "$@"\n')
            wrapper.chmod(0o755)
            # Presence checks need these names, but the failure precedes their execution.
            for name in ('git', 'g++', 'ffmpeg', 'ffprobe'):
                path = bin_dir / name
                path.write_text('#!/bin/sh\nexit 0\n')
                path.chmod(0o755)
            env = dict(os.environ, CUDA_HOME=str(root), FLASHVSR_PYTHON=str(wrapper),
                       FLASHVSR_TEST_PIP_LOG=str(log), FLASHVSR_TEST_REAL_PYTHON=sys.executable,
                       PATH=str(bin_dir) + os.pathsep + os.environ['PATH'])
            result = subprocess.run(['bash', str(PROJECT / 'scripts/install.sh')], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn('Toolkit 12.4', result.stderr)
            self.assertFalse(log.exists())


if __name__ == '__main__':
    unittest.main()
