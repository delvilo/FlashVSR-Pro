"""A CPU-only run must never produce a successful GPU acceptance report."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]


class GPUAcceptanceTests(unittest.TestCase):
    def test_no_gpu_records_failure_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([
                sys.executable, str(PROJECT / 'scripts/validate_gpu.py'), '--output-dir', temp,
            ], env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            reports = list(Path(temp).glob('run-*/report.json'))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['results'], [])
            self.assertIn('CUDA is unavailable', report['error'])


if __name__ == '__main__':
    unittest.main()
