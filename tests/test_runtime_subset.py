"""Guard the retained dependency closure and exact FlashVSR DiT architecture."""
import ast
from pathlib import Path
import subprocess
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]


class RuntimeSubsetTests(unittest.TestCase):
    def test_runtime_has_no_wildcards_or_unrelated_dependencies(self):
        banned = {'transformers', 'modelscope', 'peft', 'datasets', 'pandas', 'cv2', 'torchvision', 'torchaudio', 'xfuser'}
        for folder in ('diffsynth', 'flashvsr', 'utils'):
            for path in (PROJECT / folder).rglob('*.py'):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.ImportFrom):
                        self.assertFalse(any(item.name == '*' for item in node.names), path)
                        self.assertNotIn((node.module or '').split('.')[0], banned, path)
                    elif isinstance(node, ast.Import):
                        self.assertFalse({name.name.split('.')[0] for name in node.names} & banned, path)
        self.assertFalse((PROJECT / 'diffsynth/tokenizer_configs').exists())

    def test_all_modes_import_and_dit_shape_matches_pinned_weight_header(self):
        # Only the compiled attention function is replaced; no CUDA math is executed.
        script = '''
import sys, types
backend = types.ModuleType("block_sparse_attn")
def unavailable(*args, **kwargs):
    raise RuntimeError("GPU inference is not available in CPU tests")
backend.block_sparse_attn_func = unavailable
sys.modules["block_sparse_attn"] = backend
from diffsynth import FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
from diffsynth.models.wan_video_dit import WanModel
from diffsynth.models.utils import init_weights_on_device, hash_state_dict_keys
from flashvsr.models import ModelRegistry
release = ModelRegistry().release
with init_weights_on_device():
    model = WanModel(**release["architecture"])
assert hash_state_dict_keys(model.state_dict()) == release["state_dict_shape_hash"]
for cls in (FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline):
    pipe = cls(device="cpu")
    assert pipe.dit is None
assert "transformers" not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=PROJECT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
