"""Exercise actual PyTorch/safetensors deserialization and strict weight checks."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from utils.weights import read_checkpoint, load_checked_state_dict
from utils.vae_manager import TCDVAELoader


class WeightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_valid_checkpoint_formats(self):
        state = torch.nn.Linear(2, 2).state_dict()
        for name in ('weights.pth', 'wrapped.ckpt', 'weights.safetensors'):
            with self.subTest(name=name):
                path = self.root / name
                if path.suffix == '.safetensors':
                    save_file(state, str(path))
                else:
                    torch.save({'state_dict': state} if name.startswith('wrapped') else state, path)
                model = torch.nn.Linear(2, 2)
                load_checked_state_dict(model, read_checkpoint(path), path)
                self.assertTrue(torch.equal(model.weight, state['weight']))

    def test_invalid_or_partial_weights_stop(self):
        path = self.root / 'broken.ckpt'
        for content in (b'not a checkpoint', b''):
            path.write_bytes(content)
            with self.assertRaises((RuntimeError, FileNotFoundError)):
                read_checkpoint(path)
        torch.save({}, path)
        with self.assertRaisesRegex(RuntimeError, 'nonempty'):
            read_checkpoint(path)
        model = torch.nn.Linear(2, 2)
        for state in ({'weight': torch.ones(2, 2)}, {'weight': torch.ones(3, 2), 'bias': torch.ones(2)},
                      {**model.state_dict(), 'unknown': torch.ones(1)}):
            with self.subTest(keys=list(state)), self.assertRaisesRegex(RuntimeError, 'Incompatible model weights'):
                load_checked_state_dict(model, state, path)

    def test_tcdecoder_does_not_return_random_weights_after_load_failure(self):
        path = self.root / 'partial.ckpt'
        torch.save({'weight': torch.ones(2, 2)}, path)
        with patch('utils.vae_manager.build_tcdecoder', return_value=torch.nn.Linear(2, 2)):
            with self.assertRaisesRegex(RuntimeError, 'missing keys'):
                TCDVAELoader.load_vae(str(path), 'tcd')


if __name__ == '__main__':
    unittest.main()
