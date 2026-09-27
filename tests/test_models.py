"""Real byte verification and atomic downloads without large model weights."""
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flashvsr.models import ModelFile, ModelRegistry, model_directory, sha256_file


class ModelRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = ModelRegistry(directory=self.root)
        self.data = b"complete verified model"
        self.file = ModelFile("fixture.ckpt", len(self.data), hashlib.sha256(self.data).hexdigest(), ("tiny", "tiny-long"))
        self.registry.entries = (self.file,)

    def test_download_pins_revision_and_reuses_verified_file(self):
        with patch("urllib.request.urlopen", return_value=io.BytesIO(self.data)) as network:
            self.registry.download("tiny")
            self.registry.download("tiny-long")
        network.assert_called_once()
        self.assertIn(self.registry.release["revision"], network.call_args.args[0])
        self.assertEqual((self.root / self.file.name).read_bytes(), self.data)
        self.assertEqual(self.registry.report("tiny"), [{"name": self.file.name, "status": "ok"}])
        self.assertEqual(self.registry.selected("full"), ())

    def test_missing_truncated_lfs_and_same_size_corrupt_file_stop(self):
        path = self.root / self.file.name
        with self.assertRaisesRegex(FileNotFoundError, "Model weight missing"):
            self.registry.check("tiny")
        for data in (b"", b"partial", b"version https://git-lfs.github.com/spec/v1", b"x" * len(self.data)):
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                self.registry.check("tiny")
        path.write_bytes(self.data)
        self.registry.check("tiny")
        path.write_bytes(b"x" * len(self.data))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.registry.check("tiny")

    def test_failed_download_keeps_existing_file_and_removes_partial(self):
        path = self.root / self.file.name
        path.write_bytes(b"old file")
        for data in (b"partial", b"x" * len(self.data), self.data * 2):
            with self.subTest(data=data), patch("urllib.request.urlopen", return_value=io.BytesIO(data)):
                with self.assertRaises(ValueError):
                    self.registry.download("tiny")
            self.assertEqual(path.read_bytes(), b"old file")
            self.assertEqual(list(self.root.glob("*.part")), [])

    def test_interrupted_download_cleans_temporary_file(self):
        source = io.BytesIO(self.data)
        with patch("urllib.request.urlopen", return_value=source), patch.object(source, "read", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.registry.download("tiny")
        self.assertFalse((self.root / self.file.name).exists())
        self.assertEqual(list(self.root.glob("*.part")), [])

    def test_bundled_prompt_is_verified_and_copied_to_same_cache(self):
        registry = ModelRegistry(directory=self.root)
        registry.entries = tuple(item for item in registry.entries if item.bundled)
        with patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")):
            registry.download("full")
        self.assertEqual(sha256_file(self.root / "posi_prompt.pth"), registry.entries[0].sha256)

    def test_cache_precedence_and_manifest_modes(self):
        with patch.dict(os.environ, {"FLASHVSR_CACHE_DIR": str(self.root), "FLASHVSR_MODEL_PATH": str(self.root / "override")}, clear=True):
            self.assertEqual(model_directory(), self.root / "override")
            self.assertEqual(model_directory(directory=self.root / "explicit"), self.root / "explicit")
            del os.environ["FLASHVSR_MODEL_PATH"]
            self.assertEqual(model_directory(), self.root / "v1.1")
        with patch.dict(os.environ, {"XDG_CACHE_HOME": str(self.root)}, clear=True):
            self.assertEqual(model_directory(), self.root / "flashvsr/v1.1")
        manifest = ModelRegistry()
        full = {item.name for item in manifest.selected("full")}
        tiny = {item.name for item in manifest.selected("tiny")}
        self.assertEqual(full-tiny, {"Wan2.1_VAE.pth"})
        self.assertEqual(tiny-full, {"TCDecoder.ckpt"})
        self.assertEqual(tiny, {item.name for item in manifest.selected("tiny-long")})
        with self.assertRaises(ValueError):
            ModelRegistry("unknown")
