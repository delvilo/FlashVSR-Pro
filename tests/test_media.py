"""Real FFmpeg integration, with injected hardware/encoding failures."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from utils.media import MediaError, MediaTools
from utils.audio_utils import extract_audio, merge_audio_video


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class MediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="flashvsr media '")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.media = MediaTools()
        self.frames = [np.full((96, 128, 3), value, dtype=np.uint8) for value in range(0, 240, 30)]
        self.source = self.root / 'source.mp4'
        self.media.run(['-f', 'lavfi', '-i', 'color=size=128x96:rate=8:duration=1',
                        '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1',
                        '-f', 'lavfi', '-i', 'sine=frequency=880:duration=1',
                        '-map', '0:v', '-map', '1:a', '-map', '2:a',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', self.source])

    def test_software_encoding_preserves_all_audio_tracks_and_frames(self):
        output = self.root / 'enhanced.mp4'
        with patch.object(self.media, 'nvenc_works', return_value=False):
            info = self.media.save_video(self.frames, output, fps=8, audio_source=self.source)
        self.assertEqual((info['width'], info['height'], info['frames'], info['audio_streams']), (128, 96, 8, 2))
        self.assertAlmostEqual(info['fps'], 8)
        silent = self.root / 'silent.mp4'
        with patch.object(self.media, 'nvenc_works', return_value=False):
            self.assertFalse(self.media.save_video(self.frames, silent, fps=8)['has_audio'])
        # Requesting audio from a genuinely silent source remains a successful silent video.
        with patch.object(self.media, 'nvenc_works', return_value=False):
            self.assertEqual(self.media.save_video(self.frames, self.root / 'still-silent.mp4', fps=8,
                                                  audio_source=silent)['audio_streams'], 0)

    def test_actual_nvenc_failure_retries_software_encoding(self):
        original = self.media._pipe_frames
        codecs = []
        def encode(frames, output, fps, codec, *args):
            codecs.append(codec)
            if codec == 'h264_nvenc':
                Path(output).write_bytes(b'partial hardware output')
                raise MediaError('driver unavailable')
            return original(frames, output, fps, codec, *args)
        with patch.object(self.media, 'nvenc_works', return_value=True), patch.object(self.media, '_pipe_frames', side_effect=encode):
            with self.assertLogs('flashvsr.media', level='WARNING'):
                self.media.save_video(self.frames, self.root / 'fallback.mp4', fps=8)
        self.assertEqual(codecs, ['h264_nvenc', 'libx264'])
        self.media.verify_video(self.root / 'fallback.mp4', frames=8)

    def test_failed_encode_or_verification_preserves_previous_output(self):
        output = self.root / 'existing.mp4'
        shutil.copyfile(self.source, output)
        original = output.read_bytes()
        names = set(self.root.iterdir())
        def invalid(*args):
            Path(args[1]).write_bytes(b'not a video')
        for failure in (MediaError('encoder failed'), invalid):
            with self.subTest(failure=failure), patch.object(self.media, 'nvenc_works', return_value=False):
                with patch.object(self.media, '_pipe_frames', side_effect=failure):
                    with self.assertRaises(MediaError):
                        self.media.save_video(self.frames, output, fps=8)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(set(self.root.iterdir()), names)

    def test_probe_and_output_contract_reject_invalid_media(self):
        broken = self.root / 'bad.mp4'
        broken.write_bytes(b'broken')
        with self.assertRaises(MediaError):
            self.media.verify_video(broken)
        with self.assertRaisesRegex(MediaError, 'expected 9'):
            self.media.verify_video(self.source, frames=9)
        with self.assertRaisesRegex(MediaError, 'audio_streams'):
            self.media.verify_video(self.source, audio_streams=1)
        data = self.media.probe(self.source, count_frames=True)
        data['streams'][0]['nb_read_frames'] = 'N/A'
        with patch.object(self.media, 'probe', return_value=data):
            with self.assertRaisesRegex(MediaError, 'numeric video metadata'):
                self.media.verify_video(self.source)
        with patch.object(self.media, 'run', side_effect=MediaError('no NVENC device')):
            self.assertFalse(self.media.nvenc_works())

    def test_concat_handles_quoted_paths_and_keeps_audio_tracks(self):
        quoted = self.root / "quoted'clip.mp4"
        shutil.copyfile(self.source, quoted)
        output = self.root / 'merged.mp4'
        info = self.media.concat_videos([self.source, quoted], output)
        self.assertEqual((info['frames'], info['audio_streams']), (16, 2))

    def test_odd_resolution_uses_software_encoder(self):
        frames = [frame[:95, :127] for frame in self.frames]
        with patch.object(self.media, 'nvenc_works') as nvenc:
            info = self.media.save_video(frames, self.root / 'odd.mp4', fps=8)
            nvenc.assert_not_called()
        self.assertEqual((info['width'], info['height']), (127, 95))

    def test_other_output_containers(self):
        for suffix in ('.webm', '.gif', '.mkv', '.mov', '.avi'):
            with self.subTest(suffix=suffix), patch.object(self.media, 'nvenc_works', return_value=False):
                info = self.media.save_video(self.frames, self.root / ('output' + suffix), fps=8,
                                             audio_source=self.source if suffix == '.webm' else None)
                self.assertEqual(info['frames'], 8)
                self.assertEqual(info['audio_streams'], 2 if suffix == '.webm' else 0)

    def test_audio_helpers_preserve_multiple_tracks(self):
        audio, success = extract_audio(self.source, self.root / 'tracks.mka')
        self.assertTrue(success)
        silent = self.root / 'silent.mp4'
        with patch.object(self.media, 'nvenc_works', return_value=False):
            self.media.save_video(self.frames, silent, fps=8)
        output = self.root / 'muxed.mp4'
        self.assertTrue(merge_audio_video(silent, audio, output))
        self.media.verify_video(output, frames=8, audio_streams=2)


if __name__ == '__main__':
    unittest.main()
