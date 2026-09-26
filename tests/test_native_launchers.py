"""Exercise launcher subprocesses without downloading or running the AI models.

Run with: python -m unittest discover -s tests -v
The inference stand-in copies frames unchanged; FFmpeg splitting/merging is real.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
INFERENCE_STAND_IN = '''
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-i')
parser.add_argument('-o')
args, extra = parser.parse_known_args()
source = Path(args.i)
destination = Path(args.o) / ('processed_' + source.name)
if Path(args.o).suffix:
    destination = Path(args.o)
shutil.copyfile(source, destination)
with open(os.environ['FLASHVSR_TEST_LOG'], 'a') as log:
    log.write(json.dumps({'python': sys.executable, 'input': str(source),
                          'output': str(destination), 'extra': extra}) + '\\n')
'''


class NativeLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="flashvsr test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkout = self.root / "project with spaces"
        self.caller = self.root / "caller"
        self.bin_dir = self.root / "bin"
        for directory in (self.checkout, self.caller, self.bin_dir):
            directory.mkdir()
        (self.checkout / "infer.py").write_text(INFERENCE_STAND_IN)
        for name in ("batch_inference.py", "long_video_worker.py"):
            shutil.copyfile(PROJECT / name, self.checkout / name)
        utils = self.checkout / "utils"
        utils.mkdir()
        for name in ("__init__.py", "cli.py", "media.py"):
            shutil.copyfile(PROJECT / "utils" / name, utils / name)
        self.log = self.root / "inference.jsonl"
        self.env = dict(os.environ, FLASHVSR_TEST_LOG=str(self.log), PATH=str(self.bin_dir))

    def run_launcher(self, name, *args):
        # PATH deliberately has no `python`: children must use this interpreter.
        result = subprocess.run(
            [sys.executable, str(self.checkout / name), *map(str, args)],
            cwd=self.caller, env=self.env, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_batch_uses_checkout_paths_and_current_python(self):
        inputs = self.checkout / "inputs"
        (inputs / "nested folder").mkdir(parents=True)
        sample = inputs / "nested folder" / "example_audio.mp4"
        sample.write_bytes(b"test video")
        for excluded in ("MyOwnSwordsman_S01E01.mp4", "MyOwnSwordsman_S01E01_1080p.mp4"):
            (inputs / excluded).write_bytes(b"excluded")
        (inputs / "ignore.txt").write_text("not a video")

        calls = self.run_launcher("batch_inference.py")

        self.assertEqual(len(calls), 1)
        self.assertEqual(os.path.normpath(calls[0]["python"]), os.path.normpath(sys.executable))
        self.assertEqual(Path(calls[0]["input"]), sample)
        self.assertIn("--keep-audio", calls[0]["extra"])
        output = self.checkout / "results/nested folder/processed_example_audio.mp4"
        self.assertEqual(output.read_bytes(), sample.read_bytes())
        self.assertFalse((self.caller / "results").exists())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_long_worker_splits_and_merges_from_another_directory(self):
        ffmpeg = shutil.which("ffmpeg")
        (self.bin_dir / "ffmpeg").symlink_to(ffmpeg)
        (self.bin_dir / "ffprobe").symlink_to(shutil.which("ffprobe"))
        sample = self.caller / "sample video.mp4"
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
            "-i", "testsrc=size=128x96:rate=8", "-t", "2", "-c:v", "mpeg4",
            "-g", "8", "-pix_fmt", "yuv420p", str(sample),
        ], check=True, capture_output=True)

        calls = self.run_launcher(
            "long_video_worker.py", "-i", sample.name, "-o", "output folder",
            "--segment_time", "00:00:01",
            "--scale", "1.0",
        )

        self.assertEqual(len(calls), 2)
        self.assertTrue(all(os.path.normpath(call["python"]) == os.path.normpath(sys.executable) for call in calls))
        self.assertTrue(all("--keep-audio" in call["extra"] for call in calls))
        output = self.caller / "output folder/FlashVSR_sample video_Final.mp4"
        probe = subprocess.run([
            shutil.which("ffprobe"), "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames,width,height",
            "-of", "json", str(output),
        ], check=True, text=True, capture_output=True)
        stream = json.loads(probe.stdout)["streams"][0]
        self.assertEqual((stream["width"], stream["height"]), (128, 96))
        self.assertEqual(int(stream["nb_read_frames"]), 16)

        # A subsequent child failure must preserve the final output and clean its own work directory.
        previous = output.read_bytes()
        (self.checkout / "infer.py").write_text("raise SystemExit(9)\n")
        result = subprocess.run([
            sys.executable, str(self.checkout / "long_video_worker.py"),
            "-i", sample.name, "-o", "output folder", "--segment_time", "00:00:01",
            "--scale", "1.0",
        ], cwd=self.caller, env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        temporary = next(line.split(": ", 1)[1] for line in result.stdout.splitlines()
                         if line.startswith("Temporary directory:"))
        self.assertFalse(Path(temporary).exists())
        self.assertEqual(output.read_bytes(), previous)

    def test_batch_propagates_child_failure(self):
        inputs = self.checkout / "inputs"
        inputs.mkdir()
        (inputs / "bad.mp4").write_bytes(b"invalid")
        (self.checkout / "infer.py").write_text("raise SystemExit(7)\n")
        result = subprocess.run([sys.executable, str(self.checkout / "batch_inference.py")],
                                env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("1 failed", result.stdout)

    def test_worker_rejects_invalid_arguments_before_creating_work(self):
        result = subprocess.run([sys.executable, str(self.checkout / "long_video_worker.py"),
                                 "-i", "missing.mp4", "-o", "out", "--segment-time", "0"],
                                cwd=self.caller, env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(list(self.caller.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
