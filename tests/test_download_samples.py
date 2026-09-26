"""Check sample integrity and safe retries without downloading large videos."""

from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "download_samples", PROJECT / "scripts/download_samples.py"
)
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class SampleDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="flashvsr samples ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = b"sample video bytes"
        self.sample = {
            "bytes": len(self.data),
            "sha256": hashlib.sha256(self.data).hexdigest(),
        }
        self.name = "nested folder/example.mp4"
        self.base_url = "https://example.test/pinned/inputs/"
        self.destination = self.root / self.name
        self.output = io.StringIO()
        self.stdout = redirect_stdout(self.output)
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def download(self, **kwargs):
        return downloader.download_sample(
            self.base_url, self.name, self.sample, self.root, **kwargs
        )

    def test_verified_download_and_cache_hit(self):
        with patch.object(downloader, "urlopen", return_value=io.BytesIO(self.data)) as request:
            result = self.download()
            self.assertEqual(result, self.destination)
            self.assertEqual(result.read_bytes(), self.data)
            request.assert_called_once_with(
                self.base_url + "nested%20folder/example.mp4", timeout=60
            )
        with patch.object(downloader, "urlopen") as request:
            self.download()
            request.assert_not_called()
        self.assertEqual(list(self.root.rglob("*.part")), [])

    def test_invalid_downloads_leave_no_destination_or_partial_file(self):
        for payload in (self.data[:-1], self.data + b"extra", b"x" * len(self.data)):
            with self.subTest(payload=payload):
                with patch.object(downloader, "urlopen", return_value=io.BytesIO(payload)):
                    with self.assertRaises(ValueError):
                        self.download()
                self.assertFalse(self.destination.exists())
                self.assertEqual(list(self.root.rglob("*.part")), [])

    def test_existing_file_requires_force_and_survives_failed_replacement(self):
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(b"my original video")
        with patch.object(downloader, "urlopen") as request:
            with self.assertRaises(FileExistsError):
                self.download()
            request.assert_not_called()
        with patch.object(downloader, "urlopen", return_value=io.BytesIO(b"bad")):
            with self.assertRaises(ValueError):
                self.download(force=True)
        self.assertEqual(self.destination.read_bytes(), b"my original video")
        self.assertEqual(list(self.root.rglob("*.part")), [])
        with patch.object(downloader, "urlopen", return_value=io.BytesIO(self.data)):
            self.download(force=True)
        self.assertEqual(self.destination.read_bytes(), self.data)

    def test_interrupted_stream_is_cleaned_up_and_can_be_retried(self):
        class InterruptedStream(io.BytesIO):
            def read(self, size=-1):
                if self.tell():
                    raise URLError("connection lost")
                return super().read(3)

        with patch.object(downloader, "urlopen", return_value=InterruptedStream(self.data)):
            with self.assertRaises(URLError):
                self.download()
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.root.rglob("*.part")), [])
        with patch.object(downloader, "urlopen", return_value=io.BytesIO(self.data)):
            self.download()
        self.assertEqual(self.destination.read_bytes(), self.data)

    def test_cli_selection_defaults_and_rejects_unknown_paths_without_network(self):
        manifest = {
            "base_url": self.base_url,
            "samples": {"example0.mp4": self.sample, self.name: self.sample},
        }
        with patch.object(downloader, "load_manifest", return_value=manifest):
            with patch.object(downloader, "urlopen", return_value=io.BytesIO(self.data)) as request:
                self.assertEqual(downloader.main(["--output-dir", str(self.root)]), 0)
                request.assert_called_once_with(self.base_url + "example0.mp4", timeout=60)
                self.assertEqual((self.root / "example0.mp4").read_bytes(), self.data)
            with patch.object(downloader, "urlopen") as request:
                self.assertEqual(downloader.main(["--list"]), 0)
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    downloader.main(["example0.mp4", "unknown.mp4"])
                self.assertEqual(error.exception.code, 2)
                request.assert_not_called()

    def test_manifest_rejects_paths_outside_output_directory(self):
        path = self.root / "manifest.json"
        for name in ("../outside.mp4", "/absolute.mp4", "nested/../../outside.mp4"):
            with self.subTest(name=name):
                path.write_text(json.dumps({"version": 1, "samples": {name: self.sample}}))
                with self.assertRaises(ValueError):
                    downloader.load_manifest(path)


if __name__ == "__main__":
    unittest.main()
