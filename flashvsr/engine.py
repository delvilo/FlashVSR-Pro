"""Shared execution core with optional model reuse and isolated per-video caches."""

import gc
from contextlib import ExitStack, contextmanager
import logging
import os
import random

from .config import InferenceConfig, output_path, validate_input, validate_report_path
from .media import MediaTools
from .models import ModelRegistry
from .observability import RunReport, run_id, write_report

logger = logging.getLogger(__name__)


def pipeline_options(config, video, height, width, frames):
    options = dict(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1,
                   seed=config.seed, LQ_video=video, num_frames=frames, height=height, width=width,
                   is_full_block=False, if_buffer=True,
                   topk_ratio=config.sparse_ratio * 768 * 1280 / (height * width),
                   kv_ratio=config.kv_ratio, local_range=config.local_range, color_fix=config.color_fix)
    if config.tile_vae:
        size, overlap = max(32, config.tile_size // 8), max(4, config.overlap // 8)
        options.update(tiled=True, tile_size=(size, size), tile_stride=(size-overlap, size-overlap))
    return options


class InferenceEngine:
    def __init__(self, config: InferenceConfig, media=None, registry=None):
        self.config = config
        self.media = media
        self.registry = registry or ModelRegistry(config.model_version, config.model_dir)
        self._resources = None
        self._pipe = None
        self._verified = False
        self.model_loads = 0

    @contextmanager
    def session(self):
        """Keep one model resident until the job exits, including on interruption."""
        if self._resources is not None:
            raise RuntimeError("Inference sessions cannot be nested")
        self._resources = ExitStack()
        try:
            self.registry.check(self.config.mode)
            self._verified = True
            yield self
        finally:
            self._release_pipeline()
            self._resources = None
            self._verified = False

    def _release_pipeline(self):
        had_pipeline = self._pipe is not None
        self._pipe = None
        if self._resources is not None:
            self._resources.close()
        if had_pipeline:
            import torch
            gc.collect()
            torch.cuda.empty_cache()

    @contextmanager
    def _pipeline(self, report, synchronize):
        from .model_loading import load_pipeline, reset_pipeline
        with ExitStack() as fresh:
            report.data["model_reused"] = self._pipe is not None
            try:
                with report.stage("model_load", synchronize):
                    if self._pipe is not None:
                        pipe = self._pipe
                    else:
                        resources = self._resources if self._resources is not None else fresh
                        pipe = resources.enter_context(load_pipeline(self.config, self.registry))
                        self.model_loads += 1
                        if self._resources is not None:
                            self._pipe = pipe
                    reset_pipeline(pipe)
                try:
                    yield pipe
                finally:
                    reset_pipeline(pipe)
            except BaseException:
                # A failed invocation must never leave a partially initialized model
                # or decoder cache available to a later input.
                self._release_pipeline()
                self._verified = False
                raise

    def run(self, source, destination, metrics_json=None, *, input_frames=None, media_info=None,
            frame_start=0, lossless=False):
        config = self.config
        source = validate_input(source)
        destination = output_path(source, destination, config)
        weights = [self.registry.directory / item.name for item in self.registry.entries]
        metrics = validate_report_path(metrics_json or f"{destination}.json", source, destination, *weights)
        report = RunReport(config, source, destination, self.registry.identity(config.mode))
        token = run_id.set(report.data["run_id"])
        error = None
        try:
            logger.info("Starting %s: %s", config.mode, source)
            logger.debug("Parameters: %s", config.as_dict())
            with report.stage("preflight"):
                if not self._verified:
                    self.registry.check(config.mode)
                if self.media is None:
                    self.media = MediaTools()
                info = media_info if input_frames is not None else self.media.verify_video(source) if source.is_file() else None
                if input_frames is not None:
                    if info is None or len(input_frames) != info["frames"]:
                        raise ValueError("Buffered input must match its segment metadata")
                    report.data["frame_start"] = frame_start
                os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
                os.environ["IMAGEIO_FFMPEG_EXE"] = self.media.ffmpeg
                import torch
                from utils.runtime import validate_cuda
                index = validate_cuda(torch, config.device, config.dtype)
                report.data.update(device=torch.cuda.get_device_name(index), torch=torch.__version__,
                                   cuda=torch.version.cuda, ffmpeg=self.media.ffmpeg,
                                   ffmpeg_version=str(getattr(self.media, "ffmpeg_version", "unknown")),
                                   ffprobe=self.media.ffprobe)
            try:
                self._execute(source, destination, info, report, torch, input_frames, lossless)
            finally:
                report.data.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(config.device),
                                   peak_reserved_bytes=torch.cuda.max_memory_reserved(config.device))
                gc.collect()
                if self._pipe is None:
                    torch.cuda.empty_cache()
        except BaseException as exc:
            error = exc
            logger.error("Failed during %s: %s", report.data.get("stage", "startup"), exc,
                         exc_info=logger.isEnabledFor(logging.DEBUG))
            raise
        finally:
            data = report.finish(error)
            try:
                write_report(metrics, data)
                if error is None:
                    logger.info("Verified output: %s; %.2f s, %.2f inference FPS, peak VRAM %.2f GiB",
                                destination, data["total_seconds"], data["inference_fps"],
                                data["peak_allocated_bytes"] / 1024**3)
            except OSError:
                if error is None:
                    raise
                logger.exception("Could not save failure report: %s", metrics)
            finally:
                run_id.reset(token)
        return data

    def _execute(self, source, destination, info, report, torch, input_frames=None, lossless=False):
        from .frames import prepare_input_tensor, prepare_frame_tensor, tensor2video
        import numpy as np

        config = self.config
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        report.data["backend"] = {"allow_tf32": True, "cudnn_benchmark": True}
        torch.cuda.reset_peak_memory_stats(config.device)
        synchronize = lambda: torch.cuda.synchronize(config.device)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[config.dtype]
        with self._pipeline(report, synchronize) as pipe:
            with report.stage("decode", synchronize):
                if input_frames is None:
                    prepared = prepare_input_tensor(str(source), config.scale, dtype, config.device, media_info=info)
                else:
                    prepared = prepare_frame_tensor(input_frames, len(input_frames), info["width"], info["height"],
                                                    info["fps"], config.scale, dtype, config.device)
                lq, height, width, count, fps, audio_source, original, exact_h, exact_w = prepared
                fps = config.fps or fps
            options = pipeline_options(config, lq, height, width, count)
            report.data["effective"] = {"height": height, "width": width, "padded_frames": count,
                                        "output_fps": fps, "topk_ratio": options["topk_ratio"]}
            with report.stage("inference", synchronize), torch.inference_mode():
                if config.tile_dit:
                    from utils.tile_utils import apply_tiled_inference_simple
                    options.pop("LQ_video")
                    video = apply_tiled_inference_simple(
                        pipe, lq, tile_size=config.tile_size, overlap=config.overlap,
                        tile_size_vae=options.pop("tile_size", None), **options)
                else:
                    video = pipe(**options)
            with report.stage("postprocess", synchronize):
                h, w = video.shape[-2:]
                if h < exact_h or w < exact_w:
                    raise RuntimeError("Pipeline returned smaller dimensions than requested")
                video = video[..., (h-exact_h)//2:(h-exact_h)//2+exact_h, (w-exact_w)//2:(w-exact_w)//2+exact_w]
                frames = tensor2video(video)
                if not frames:
                    raise RuntimeError("Pipeline returned no frames")
                report.data["generated_frames"] = len(frames)
                if len(frames) < original:
                    logger.warning("Pipeline returned %s/%s frames; repeating the last frame", len(frames), original)
                    frames.extend([frames[-1]] * (original-len(frames)))
                frames = frames[:original]
            with report.stage("encode"):
                output = self.media.save_video(frames, destination, fps=fps, quality=config.quality,
                                               audio_source=audio_source if config.keep_audio else None,
                                               device_index=torch.cuda.current_device(), lossless=lossless)
            report.data.update(output)
            report.data.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(config.device),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved(config.device))
