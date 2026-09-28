"""Sequential FFmpeg decoding with one caller-sized RGB buffer and backpressure."""

from contextlib import contextmanager
import subprocess
import tempfile

from .media import MediaError


class FrameReader:
    def __init__(self, process, errors, width, height):
        self.process, self.errors = process, errors
        self.width, self.height = width, height
        self.frame_bytes = width * height * 3

    def read(self, count):
        import numpy as np
        frames = np.empty((count, self.height, self.width, 3), dtype=np.uint8)
        view = memoryview(frames).cast('B')
        offset = 0
        while offset < len(view):
            size = self.process.stdout.readinto(view[offset:])
            if not size:
                raise MediaError("Video decoder ended before the expected frame count")
            offset += size
        return frames

    def discard(self, count):
        remaining = count * self.frame_bytes
        while remaining:
            data = self.process.stdout.read(min(remaining, 1024 * 1024))
            if not data:
                raise MediaError("Video decoder ended while skipping completed frames")
            remaining -= len(data)

    def finish(self):
        if self.process.stdout.read(1):
            raise MediaError("Video decoder produced more frames than FFprobe counted")
        code = self.process.wait(timeout=120)
        self.errors.seek(0)
        detail = self.errors.read(4000).decode('utf-8', errors='replace').strip()
        if code or detail:
            raise MediaError(f"Video decoder failed (exit {code}): {detail}")


@contextmanager
def read_video_frames(media, source, info, start_frame=0):
    """Resume by decoded frame index, independent of keyframe spacing or timestamps.

    FFmpeg scans a completed prefix when seeking is unsafe, but does not send it
    through Python or run inference on it. Pipe backpressure bounds read-ahead.
    """
    arguments = [media.ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
                 '-noautorotate', '-i', str(source), '-map', '0:v:0', '-an', '-sn', '-dn']
    if start_frame:
        arguments += ['-vf', f'select=gte(n\\,{start_frame})']
    arguments += (['-fps_mode', 'passthrough'] if media.supports_fps_mode else ['-vsync', '0'])
    arguments += ['-threads', '1', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1']
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=errors)
        try:
            reader = FrameReader(process, errors, info['width'], info['height'])
            yield reader
            reader.finish()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
