"""Compatibility helpers using the shared, verified FFmpeg output path.

Failures raise MediaError instead of silently returning a video without audio.
"""

import os
from pathlib import Path
import tempfile

from .media import MediaError, MediaTools, atomic_output


def has_audio_stream(video_path):
    return MediaTools().has_audio(video_path)


def extract_audio(video_path, audio_output_path=None):
    tools = MediaTools()
    if not tools.has_audio(video_path):
        raise MediaError(f'No audio stream in {video_path}')
    generated = audio_output_path is None
    if generated:
        descriptor, audio_output_path = tempfile.mkstemp(prefix='flashvsr-audio-', suffix='.mka')
        os.close(descriptor)
    try:
        with atomic_output(audio_output_path) as temporary:
            tools.run(['-i', video_path, '-map', '0:a', '-vn', '-c:a', 'copy', temporary])
            if not tools.has_audio(temporary):
                raise MediaError('Extracted output has no audio stream')
        return str(audio_output_path), True
    except BaseException:
        if generated:
            Path(audio_output_path).unlink(missing_ok=True)
        raise


def merge_audio_video(video_path, audio_path, output_path):
    MediaTools().mux_audio(video_path, audio_path, output_path)
    return True


def copy_video_with_audio(original_video_path, processed_video_path, output_path):
    MediaTools().mux_audio(processed_video_path, original_video_path, output_path)
    return True
