import logging
logger = logging.getLogger(__name__)

"""FFmpeg encoding, probing and atomic output shared by the native programs."""

from contextlib import contextmanager
from functools import lru_cache
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


class MediaError(RuntimeError):
    """A media operation failed; no successful output may be reported."""


@contextmanager
def atomic_output(destination):
    """Keep the previous destination until the caller verifies the temporary file."""
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=destination.suffix, dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise MediaError(f"Output is missing or empty: {temporary}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _executable(value, fallback):
    resolved = shutil.which(str(value or fallback))
    if not resolved:
        raise MediaError(f"Required executable not found: {value or fallback}. See INSTALLATION.md.")
    return str(Path(resolved).resolve())


def _rate(value):
    try:
        number = float(Fraction(value))
        return number if math.isfinite(number) and number > 0 else 0.0
    except (ValueError, ZeroDivisionError, TypeError):
        return 0.0


class MediaTools:
    def __init__(self, ffmpeg=None, ffprobe=None):
        self.ffmpeg = _executable(ffmpeg or os.getenv("FLASHVSR_FFMPEG"), "ffmpeg")
        self.ffprobe = _executable(ffprobe or os.getenv("FLASHVSR_FFPROBE"), "ffprobe")
        try:
            result = subprocess.run([self.ffmpeg, "-version"], capture_output=True, text=True,
                                    check=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as error:
            raise MediaError(f"Cannot determine FFmpeg version: {error}") from error
        version_output = result.stdout or result.stderr
        self.ffmpeg_version = version_output.splitlines()[0].strip()
        match = re.search(r"\bversion\s+n?(\d+)\.(\d+)(?:\.(\d+))?", self.ffmpeg_version)
        self.ffmpeg_release = tuple(map(int, match.groups(default="0"))) if match else None
        # -fps_mode was introduced in 5.1; older releases use the equivalent
        # -vsync numeric mode. Keep the selected executable and its options paired.
        if self.ffmpeg_release is None:
            try:
                help_result = subprocess.run([self.ffmpeg, "-hide_banner", "-h", "full"],
                                             capture_output=True, text=True, timeout=10)
                self.supports_fps_mode = "-fps_mode" in (help_result.stdout + help_result.stderr)
            except (OSError, subprocess.SubprocessError):
                self.supports_fps_mode = False
        else:
            self.supports_fps_mode = self.ffmpeg_release >= (5, 1, 0)
        logger.info("Using %s", self.ffmpeg_version)

    def cfr_arguments(self):
        """Return the constant-frame-rate option supported by this FFmpeg."""
        return ["-fps_mode", "cfr"] if self.supports_fps_mode else ["-vsync", "1"]

    def run(self, arguments, timeout=None):
        try:
            result = subprocess.run(
                [self.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *map(str, arguments)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise MediaError(f"FFmpeg failed: {error}") from error
        if result.returncode:
            raise MediaError(f"FFmpeg exited {result.returncode}: {result.stderr[-4000:].strip()}")

    def probe(self, path, count_frames=False):
        command = [self.ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json"]
        if count_frames:
            command.append("-count_frames")
        command.append(str(path))
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            if result.stderr.strip():
                raise MediaError(f"Invalid media {path}: {result.stderr[-4000:].strip()}")
            return json.loads(result.stdout)
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            detail = getattr(error, "stderr", None) or str(error)
            raise MediaError(f"Cannot probe {path}: {detail[-4000:]}") from error

    def has_audio(self, path):
        return any(stream.get("codec_type") == "audio" for stream in self.probe(path).get("streams", []))

    def verify_video(self, path, *, width=None, height=None, frames=None, audio=None,
                     audio_streams=None, fps=None, video_codec=None):
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise MediaError(f"Output is missing or empty: {path}")
        data = self.probe(path, count_frames=True)
        videos = [stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"]
        if not videos:
            raise MediaError(f"No video stream found in {path}")
        stream = videos[0]
        try:
            actual = {
                "width": int(stream.get("width", 0)),
                "height": int(stream.get("height", 0)),
                "frames": int(stream.get("nb_read_frames", 0) or 0),
                "video_codec": stream.get("codec_name"),
                "fps": _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate")),
                "has_audio": any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
                "audio_streams": sum(s.get("codec_type") == "audio" for s in data.get("streams", [])),
                "duration": float(data.get("format", {}).get("duration", 0) or 0),
            }
        except (TypeError, ValueError) as error:
            raise MediaError(f"Invalid numeric video metadata: {path}") from error
        if actual["fps"] <= 0 or not math.isfinite(actual["duration"]):
            raise MediaError(f"Invalid video timing metadata: {path}")
        if min(actual["width"], actual["height"], actual["frames"]) <= 0:
            raise MediaError(f"Invalid video dimensions or frame count: {path}")
        for key, expected in (("width", width), ("height", height), ("frames", frames),
                              ("has_audio", audio), ("audio_streams", audio_streams),
                              ("video_codec", video_codec)):
            if expected is not None and actual[key] != expected:
                raise MediaError(f"Output {key}: expected {expected}, got {actual[key]} ({path})")
        if fps is not None and not math.isclose(actual["fps"], fps, rel_tol=0.005, abs_tol=0.01):
            raise MediaError(f"Output FPS: expected {fps}, got {actual['fps']} ({path})")
        return actual

    @staticmethod
    def _hevc_nvenc_arguments(quality, device_index):
        # The default quality 10 maps to CQ 20; lower quality selects a larger CQ.
        cq = str(int(26 - quality * 0.6))
        return ["-gpu", str(device_index), "-preset", "p7", "-tune", "hq",
                "-rc", "vbr", "-cq", cq, "-b:v", "0", "-multipass", "fullres",
                "-bf", "3", "-b_ref_mode", "middle", "-rc-lookahead", "32",
                "-spatial-aq", "1", "-temporal-aq", "1"]

    @lru_cache(maxsize=16)
    def nvenc_works(self, device_index=0, quality=10):
        # Exercise the full preset, including more frames than the lookahead window.
        try:
            self.run([
                "-f", "lavfi", "-i", "color=size=128x128:rate=30:duration=2",
                "-map", "0:v:0", "-c:v", "hevc_nvenc",
                *self._hevc_nvenc_arguments(quality, device_index),
                "-pix_fmt", "yuv420p", *self.cfr_arguments(),
                "-frames:v", "40", "-f", "null", "-",
            ], timeout=30)
            return True
        except MediaError as error:
            logger.debug("HEVC NVENC test encode unavailable: %s", error)
            return False

    @lru_cache(maxsize=16)
    def _h264_nvenc_works(self, device_index=0):
        # AVI retains its existing H.264 path; HEVC is used for MP4/MOV/MKV.
        try:
            self.run(["-f", "lavfi", "-i", "color=size=128x128:rate=1",
                      "-frames:v", "1", "-c:v", "h264_nvenc", "-gpu", str(device_index),
                      "-f", "null", "-"], timeout=15)
            return True
        except MediaError:
            return False

    def _pipe_frames(self, frames, output, fps, codec, quality, audio_source, device_index):
        import numpy as np

        first = np.asarray(frames[0])
        if first.ndim != 3 or first.shape[2] != 3 or first.dtype != np.uint8:
            raise MediaError("Video frames must be RGB uint8 arrays")
        height, width = first.shape[:2]
        quality_value = str(int(26 - quality * 0.6))
        arguments = [
            self.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
            "-r", str(fps), "-i", "pipe:0",
        ]
        if audio_source is not None:
            arguments += ["-i", str(audio_source)]
        arguments += ["-map", "0:v:0", "-c:v", codec]
        if codec == "hevc_nvenc":
            arguments += self._hevc_nvenc_arguments(quality, device_index)
        elif codec == "h264_nvenc":
            arguments += ["-gpu", str(device_index), "-preset", "p1", "-rc", "vbr", "-cq", quality_value, "-b:v", "0"]
        elif codec == "libx264":
            arguments += ["-preset", "veryfast", "-crf", quality_value]
        elif codec == "libvpx-vp9":
            arguments += ["-deadline", "good", "-cpu-used", "4", "-crf", quality_value, "-b:v", "0"]
        if codec == "ffv1":
            arguments += ["-level", "3", "-pix_fmt", "bgr0"]
        elif codec != "gif":
            arguments += ["-pix_fmt", "yuv420p" if width % 2 == height % 2 == 0 else "yuv444p"]
        if audio_source is not None:
            arguments += [
                "-map", "1:a", "-c:a", "libopus" if output.suffix.lower() == ".webm" else "aac",
                "-b:a", "192k", "-af", "apad", "-shortest",
            ]
        else:
            arguments += ["-an"]
        if output.suffix.lower() != ".gif":
            arguments += self.cfr_arguments()
        if output.suffix.lower() in (".mp4", ".mov"):
            arguments += ["-movflags", "+faststart"]
        arguments.append(str(output))

        # A disk-backed stderr avoids a deadlock if FFmpeg writes while stdin is full.
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
            broken_pipe = False
            try:
                for frame in frames:
                    array = np.asarray(frame)
                    if array.shape != first.shape or array.dtype != np.uint8:
                        raise MediaError("Every output frame must have the same RGB uint8 shape")
                    process.stdin.write(array.tobytes())
                process.stdin.close()
            except BrokenPipeError:
                broken_pipe = True
            except BaseException:
                process.kill()
                process.wait()
                raise
            finally:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                code = process.wait(timeout=120)
            except BaseException:
                process.kill()
                process.wait()
                raise
            if code or broken_pipe:
                errors.seek(max(0, errors.seek(0, 2) - 4000))
                detail = errors.read().decode("utf-8", errors="replace").strip()
                raise MediaError(f"{codec} encoding failed (exit {code}): {detail}")

    def save_video(self, frames, destination, fps=30, quality=10, audio_source=None, device_index=0, lossless=False):
        import numpy as np

        # Frames must be replayable so a failed NVENC encode can retry on the CPU.
        if len(frames) == 0:
            raise MediaError("Cannot save a video with no frames")
        if not math.isfinite(fps) or fps <= 0 or not 0 <= quality <= 10:
            raise ValueError("FPS must be finite and positive; quality must be in 0..10")
        destination = Path(destination).resolve()
        suffix = destination.suffix.lower()
        if suffix not in {".mp4", ".mov", ".mkv", ".avi", ".webm", ".gif"}:
            raise MediaError(f"Unsupported output format: {suffix}")
        if audio_source is not None and Path(audio_source).resolve() == destination:
            raise MediaError("Output must not overwrite its audio source")
        audio_count = 0
        if audio_source is not None:
            audio_count = sum(s.get("codec_type") == "audio" for s in self.probe(audio_source).get("streams", []))
            if not audio_count:
                logger.info(f"Input has no audio tracks: {audio_source}")
                audio_source = None
        if suffix == ".gif" and audio_source is not None:
            raise MediaError("GIF cannot preserve audio")
        first = np.asarray(frames[0])
        if first.ndim != 3 or first.shape[2] != 3 or first.dtype != np.uint8:
            raise MediaError("Video frames must be RGB uint8 arrays")
        height, width = first.shape[:2]
        if lossless:
            if suffix != ".mkv" or audio_source is not None:
                raise MediaError("Lossless job segments require silent Matroska output")
            codecs = ["ffv1"]
        elif suffix == ".gif":
            codecs = ["gif"]
        elif suffix == ".webm":
            codecs = ["libvpx-vp9"]
        else:
            codecs = ["libx264"]
            if width % 2 == height % 2 == 0:
                if suffix in {".mp4", ".mov", ".mkv"} and self.nvenc_works(device_index, quality):
                    codecs.insert(0, "hevc_nvenc")
                elif suffix == ".avi" and self._h264_nvenc_works(device_index):
                    codecs.insert(0, "h264_nvenc")
        output_codecs = {"hevc_nvenc": "hevc", "h264_nvenc": "h264",
                         "libx264": "h264", "libvpx-vp9": "vp9", "gif": "gif", "ffv1": "ffv1"}
        for codec in codecs:
            try:
                with atomic_output(destination) as temporary:
                    self._pipe_frames(frames, temporary, fps, codec, quality, audio_source, device_index)
                    info = self.verify_video(
                        temporary, width=width, height=height, frames=len(frames),
                        audio_streams=audio_count, fps=fps if suffix != ".gif" else None,
                        video_codec=output_codecs[codec],
                    )
                logger.info(f"Saved and verified with {codec}: {destination}")
                return info
            except (MediaError, OSError, subprocess.SubprocessError) as error:
                if codec not in {"hevc_nvenc", "h264_nvenc"}:
                    raise MediaError(f"Cannot save {destination}: {error}") from error
                logger.warning("NVENC failed; retrying with libx264: %s", error)
        raise MediaError(f"No encoder succeeded: {destination}")

    def encode_segments(self, paths, destination, *, fps, quality=10, audio_source=None, device_index=0):
        """Encode lossless job segments once, with replayable NVENC fallback.

        Concatenation, encoding and audio muxing stream inside FFmpeg; Python
        stores only the segment metadata. Atomic publication retains old output.
        """
        if not paths:
            raise MediaError("No completed job segments")
        metadata = [self.verify_video(path, audio_streams=0, fps=fps, video_codec="ffv1") for path in paths]
        width, height = metadata[0]["width"], metadata[0]["height"]
        if any((info["width"], info["height"]) != (width, height) for info in metadata):
            raise MediaError("Segment dimensions do not match")
        frames = sum(info["frames"] for info in metadata)
        audio_count = sum(s.get("codec_type") == "audio" for s in self.probe(audio_source).get("streams", [])) if audio_source else 0
        codecs = ["libx264"]
        if width % 2 == height % 2 == 0 and self.nvenc_works(device_index, quality):
            codecs.insert(0, "hevc_nvenc")
        with tempfile.TemporaryDirectory(prefix="flashvsr-encode-") as folder:
            listing = Path(folder) / "segments.txt"
            with listing.open("w", encoding="utf-8") as stream:
                for path, info in zip(paths, metadata):
                    name = str(Path(path).resolve())
                    if "\n" in name or "\r" in name:
                        raise MediaError("Segment paths cannot contain newlines")
                    stream.write("file '" + name.replace("'", "'\\''") + "'\n")
                    stream.write(f"duration {info['frames'] / fps:.9f}\n")
            for codec in codecs:
                try:
                    with atomic_output(destination) as temporary:
                        args = ["-f", "concat", "-safe", "0", "-i", listing]
                        if audio_count:
                            args += ["-i", audio_source]
                        args += ["-map", "0:v:0", "-vf", f"setpts=N/({fps}*TB)", "-r", str(fps), "-c:v", codec]
                        if codec == "hevc_nvenc":
                            args += self._hevc_nvenc_arguments(quality, device_index)
                        else:
                            args += ["-preset", "veryfast", "-crf", str(int(26 - quality * 0.6))]
                        args += ["-pix_fmt", "yuv420p" if width % 2 == height % 2 == 0 else "yuv444p", *self.cfr_arguments()]
                        if audio_count:
                            args += ["-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-af", "apad", "-shortest"]
                        else:
                            args += ["-an"]
                        self.run([*args, "-movflags", "+faststart", temporary])
                        result = self.verify_video(temporary, width=width, height=height, frames=frames, fps=fps,
                                                   audio_streams=audio_count, video_codec="hevc" if codec == "hevc_nvenc" else "h264")
                    return result
                except (MediaError, OSError, subprocess.SubprocessError) as error:
                    if codec != "hevc_nvenc":
                        raise
                    logger.warning("Final NVENC encode failed; retrying with libx264: %s", error)

    def mux_audio(self, video, audio_source, destination):
        info = self.verify_video(video)
        audio_count = sum(s.get("codec_type") == "audio" for s in self.probe(audio_source).get("streams", []))
        destination = Path(destination)
        if destination.resolve() == Path(audio_source).resolve():
            raise MediaError("Output must not overwrite its audio source")
        with atomic_output(destination) as temporary:
            if audio_count:
                self.run([
                    "-i", video, "-i", audio_source, "-map", "0:v:0", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "libopus" if destination.suffix.lower() == ".webm" else "aac",
                    "-af", "apad", "-shortest", temporary,
                ])
            else:
                self.run(["-i", video, "-map", "0:v:0", "-c:v", "copy", "-an", temporary])
            return self.verify_video(
                temporary, width=info["width"], height=info["height"], frames=info["frames"], audio_streams=audio_count,
                fps=info["fps"], video_codec=info["video_codec"],
            )

    def concat_videos(self, paths, destination):
        if not paths:
            raise MediaError("No video segments to concatenate")
        metadata = [self.verify_video(path) for path in paths]
        reference = metadata[0]
        for info in metadata[1:]:
            if any(info[key] != reference[key] for key in ("width", "height", "audio_streams", "video_codec")):
                raise MediaError("Segment dimensions, codecs or audio streams do not match")
            if not math.isclose(info["fps"], reference["fps"], rel_tol=0.001):
                raise MediaError("Segment frame rates do not match")
        with tempfile.TemporaryDirectory(prefix="flashvsr-concat-") as folder:
            listing = Path(folder) / "segments.txt"
            lines = []
            for path, info in zip(paths, metadata):
                name = str(Path(path).resolve())
                if "\n" in name or "\r" in name:
                    raise MediaError("Segment paths cannot contain newlines")
                lines.append("file '" + name.replace("'", "'\\''") + "'\n")
                # Container duration can include an initial timestamp offset or
                # audio padding. Place the next video after the decoded frames.
                lines.append(f"duration {info['frames'] / info['fps']:.9f}\n")
            listing.write_text("".join(lines), encoding="utf-8")
            with atomic_output(destination) as temporary:
                self.run(["-f", "concat", "-safe", "0", "-i", listing, "-map", "0:v:0", "-map", "0:a?", "-c", "copy", temporary])
                return self.verify_video(
                    temporary, width=reference["width"], height=reference["height"],
                    frames=sum(item["frames"] for item in metadata), audio_streams=reference["audio_streams"],
                    fps=reference["fps"], video_codec=reference["video_codec"],
                )
