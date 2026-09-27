"""Test CPU frame preparation and reject silently corrupted decoded frames."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch
from einops import rearrange

from flashvsr import frames as infer


class InputFrameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_image_sequence_keeps_original_dimensions_and_count(self):
        for index in range(2):
            Image.new('RGB', (13, 9), (index * 100, 50, 25)).save(self.root / f'{index}.png')
        video, height, width, count, fps, source, original, exact_h, exact_w = infer.prepare_input_tensor(
            str(self.root), scale=2.0, device='cpu', dtype=torch.float32)
        self.assertEqual(video.shape, (1, 3, 9, 128, 128))
        self.assertEqual((original, exact_h, exact_w), (2, 18, 26))
        self.assertTrue(torch.equal(video[:, :, 1], video[:, :, -1]))

    def test_reader_failure_is_propagated_and_closed(self):
        source = self.root / 'broken.mp4'
        source.touch()
        reader = MagicMock()
        reader.get_data.side_effect = RuntimeError('decode failed')
        with patch.object(imageio, 'get_reader', return_value=reader):
            with self.assertRaisesRegex(RuntimeError, 'decode failed'):
                infer.prepare_input_tensor(str(source), device='cpu', dtype=torch.float32,
                                           media_info={'frames': 2, 'fps': 8})
        reader.close.assert_called_once()

    def test_nonfinite_output_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'non-finite'):
            infer.tensor2video(torch.full((3, 1, 2, 2), float('nan')))


if __name__ == '__main__':
    unittest.main()
