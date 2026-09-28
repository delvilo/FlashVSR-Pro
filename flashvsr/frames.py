"""Frame decoding, spatial padding and tensor conversion for all workflows."""
from __future__ import annotations
import logging
import os
import re
import numpy as np
import torch
from PIL import Image
import imageio.v2 as imageio
from einops import rearrange
from .media import MediaTools

logger = logging.getLogger(__name__)


def tensor2video(frames: torch.Tensor):
    """Convert tensor to list of Numpy Arrays (uint8)"""
    # Handle optional batch dimension
    if frames.ndim == 5:
        frames = frames.squeeze(0)

    if not torch.isfinite(frames).all():
        raise RuntimeError("Pipeline returned non-finite pixel values")
    frames = rearrange(frames, "C T H W -> T H W C")
    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    # Optimization: Return numpy arrays directly to avoid costly PIL conversion
    # Pipeline consumers (imageio, ffmpeg pipe) handle numpy arrays efficiently
    frames = [frame for frame in frames]
    return frames

def natural_key(name: str):
    """Natural sort key for filenames"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    """List image files with natural sorting"""
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder)
          if f.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.isfile(os.path.join(folder, f))]
    fs.sort(key=natural_key)
    return fs

def is_video(path):
    """Check if path is a video file"""
    return os.path.isfile(path) and path.lower().endswith(('.mp4','.mov','.avi','.mkv','.webm'))

def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 2.0, multiple: int = 128):
    """Compute scaled dimensions and target dimensions"""
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    sW = int(round(w0 * scale))
    sH = int(round(h0 * scale))
    if min(sW, sH) <= 0:
        raise ValueError("Scaled dimensions must be at least one pixel")

    # Pad to multiple instead of cropping (Round UP)
    # This prevents resolution loss (1480 -> 1408) by going 1480 -> 1536
    # The result will be cropped back to sW, sH after inference
    tW = ((sW + multiple - 1) // multiple) * multiple
    tH = ((sH + multiple - 1) // multiple) * multiple

    if tW == 0 or tH == 0:
        raise ValueError(
            f"Scaled size too small ({sW}x{sH}) for multiple={multiple}. "
            f"Increase scale (got {scale})."
        )

    return sW, sH, tW, tH

def process_frame_gpu(img_or_arr, sH: int, sW: int, tH: int, tW: int, dtype=None, device='cuda'):
    # Kept for backward compatibility or single frame processing
    return process_batch_gpu([img_or_arr], sH, sW, tH, tW, dtype, device).squeeze(0)

def process_batch_gpu(batch_arr, sH: int, sW: int, tH: int, tW: int, dtype=None, device='cuda'):
    """
    Process a batch of frames on GPU:
    1. Convert stacked numpy array/list to tensor
    2. Upscale (Bicubic) to (sH, sW)
    3. Center Crop to (tH, tW)
    4. Normalize to [-1, 1]
    """
    if dtype is None:
        dtype = torch.bfloat16
    if isinstance(batch_arr, list):
         # Handle list of PIL images or numpy arrays
         if len(batch_arr) > 0 and isinstance(batch_arr[0], Image.Image):
             batch_arr = np.stack([np.array(img) for img in batch_arr])
         else:
             batch_arr = np.stack(batch_arr)

    # Input: (B, H, W, C) -> Output: (B, C, H, W)
    t = torch.from_numpy(batch_arr).to(device=device, dtype=dtype)
    t = t.permute(0, 3, 1, 2) # (B, C, H, W)

    # 2. Upscale
    if t.shape[2] != sH or t.shape[3] != sW:
        t = torch.nn.functional.interpolate(t, size=(sH, sW), mode='bicubic', align_corners=False)

    # 3. Pad to Target Size (tH, tW)
    # Target size is guaranteed to be >= sW/sH by compute_scaled_and_target_dims
    curr_h, curr_w = t.shape[2], t.shape[3]
    pad_h = tH - curr_h
    pad_w = tW - curr_w

    if pad_h > 0 or pad_w > 0:
        # Pad format: (left, right, top, bottom)
        # We pad right and bottom for simplicity, or center?
        # Center padding is better for symmetric context.
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        t = torch.nn.functional.pad(t, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect' if pad_top < curr_h and pad_bottom < curr_h and pad_left < curr_w and pad_right < curr_w else 'replicate')

    # 4. Normalize [0, 255] -> [-1, 1]
    t = t / 255.0 * 2.0 - 1.0

    return t

def prepare_frame_tensor(frames, total, width, height, fps, scale=2, dtype=None, device='cuda', audio_source=None):
    """Preallocate one input tensor; retain at most 32 unscaled frames while filling it.

    Streaming pipelines need lookahead beyond the output interval, including for
    a one-frame final segment. Repeat the last input to supply that context.
    """
    if total < 1 or fps <= 0:
        raise ValueError("Frame count and FPS must be positive")
    dtype = dtype or torch.bfloat16
    sW, sH, tW, tH = compute_scaled_and_target_dims(width, height, scale=scale)
    count = max(25, ((total - 1 + 7) // 8) * 8 + 1 + 16)
    video = torch.empty((1, 3, count, tH, tW), dtype=dtype, device=device)
    batch, offset = [], 0
    for index, frame in enumerate(frames):
        if index >= total:
            raise ValueError("Decoder produced too many frames")
        if frame.shape != (height, width, 3):
            raise ValueError(f"Frame dimensions changed at frame {index}")
        batch.append(frame)
        if len(batch) == 32 or index == total - 1:
            processed = process_batch_gpu(batch, sH, sW, tH, tW, dtype, device)
            video[0, :, offset:offset+len(batch)].copy_(processed.permute(1, 0, 2, 3))
            offset += len(batch)
            batch.clear()
            del processed
    if offset != total:
        raise ValueError(f"Decoder produced {offset}/{total} frames")
    video[:, :, total:] = video[:, :, total-1:total].expand(-1, -1, count-total, -1, -1)
    return video, tH, tW, count, fps, audio_source, total, sH, sW


def prepare_input_tensor(path: str, scale: float = 2, dtype=None, device='cuda', media_info=None):
    """Single-input API; long jobs supply a bounded frame sequence directly."""
    if os.path.isdir(path):
        paths = list_images_natural(path)
        if not paths:
            raise FileNotFoundError(f"No images in {path}")
        with Image.open(paths[0]) as first:
            w0, h0 = first.size
        def images():
            for name in paths:
                with Image.open(name) as image:
                    if image.size != (w0, h0):
                        raise ValueError(f"Image dimensions differ from the first frame: {name}")
                    yield np.array(image.convert('RGB'))
        return prepare_frame_tensor(images(), len(paths), w0, h0, 30.0, scale, dtype, device)

    if not is_video(path):
        raise ValueError(f"Unsupported input: {path}")
    info = media_info if media_info is not None else MediaTools().verify_video(path)
    total, fps = info['frames'], info['fps']
    if fps <= 0:
        raise ValueError(f"Input has no valid frame rate: {path}")
    reader = imageio.get_reader(path)
    try:
        first = reader.get_data(0)
        h0, w0 = first.shape[:2]
        logger.info(f"Input: {w0}x{h0}, {total} frames, {fps:.3f} FPS")
        def frames():
            yield first
            for index in range(1, total):
                yield reader.get_data(index)
        return prepare_frame_tensor(frames(), total, w0, h0, fps, scale, dtype, device, path)
    finally:
        reader.close()
