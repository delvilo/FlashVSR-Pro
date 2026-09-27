#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import re
import sys
import time
from contextlib import redirect_stdout

from utils.cli import parse_args, validate_input, validate_output, run_cli
from utils.media import MediaTools, atomic_output
from utils.runtime import validate_cuda, validate_models
from utils.weights import read_checkpoint, load_checked_state_dict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")


def load_runtime(args, media):
    """Load heavy dependencies only after CLI and file checks have succeeded."""
    global torch, np, Image, imageio, tqdm, rearrange
    global ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
    global Causal_LQ4x_Proj, vae_manager, apply_tiled_inference_simple
    import torch
    validate_cuda(torch, args.device, args.dtype)
    # Reading and writing use the same selected FFmpeg installation.
    os.environ["IMAGEIO_FFMPEG_EXE"] = media.ffmpeg
    import numpy as np
    from PIL import Image
    import imageio.v2 as imageio
    from tqdm import tqdm
    from einops import rearrange
    from diffsynth import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
    from utils.utils import Causal_LQ4x_Proj
    from utils import vae_manager
    from utils.tile_utils import apply_tiled_inference_simple

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
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def largest_8n1_leq(n):
    """Find largest 8n+1 <= n"""
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

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
    
    return t # Output stays on GPU

def prepare_input_tensor(path: str, scale: float = 2, dtype=None, device='cuda', media_info=None):
    """Read every input frame and pad to 8n+1 without hiding decoder failures."""
    if os.path.isdir(path):
        paths = list_images_natural(path)
        if not paths:
            raise FileNotFoundError(f"No images in {path}")
        with Image.open(paths[0]) as first:
            w0, h0 = first.size
        total, fps = len(paths), 30.0
        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale)
        count = ((total - 1 + 7) // 8) * 8 + 1
        frames = []
        for name in paths + [paths[-1]] * (count - total):
            with Image.open(name) as image:
                if image.size != (w0, h0):
                    raise ValueError(f"Image dimensions differ from the first frame: {name}")
                frames.append(process_frame_gpu(image.convert('RGB'), sH, sW, tH, tW, dtype, device))
        video = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)
        return video, tH, tW, count, fps, None, total, sH, sW

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
        sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale)
        count = ((total - 1 + 7) // 8) * 8 + 1
        print(f"Input: {w0}x{h0}, {total} frames, {fps:.3f} FPS")
        frames, batch = [], []
        for index in list(range(total)) + [total - 1] * (count - total):
            frame = reader.get_data(index)
            if frame.shape[:2] != (h0, w0):
                raise ValueError(f"Frame dimensions changed in {path} at frame {index}")
            batch.append(frame)
            if len(batch) == 32:
                frames.append(process_batch_gpu(batch, sH, sW, tH, tW, dtype, device))
                batch = []
        if batch:
            frames.append(process_batch_gpu(batch, sH, sW, tH, tW, dtype, device))
        video = torch.cat(frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
        return video, tH, tW, count, fps, path, total, sH, sW
    finally:
        reader.close()


def init_pipeline(args):
    """Initialize pipeline based on mode"""
    print(f"Device: {torch.cuda.current_device()}, {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    model_dir = str(validate_models(args.mode))
    print(f"Loading models from: {model_dir}")
    
    # Setup dtype
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Initialize VAE Manager and load VAE
    # Determine vae_type automatically from mode
    vae_type = "wan2.1" if args.mode == 'full' else "tcd"
    
    # Suppress output from vae loading as we will control it
    with redirect_stdout(io.StringIO()):
        vae_system_instance = vae_manager.VAESystem(device=args.device, dtype=dtype)
        vae_model = vae_system_instance.load_vae(
            vae_type=vae_type,
            weight_path=None,  # No user custom path supported
            mode=args.mode,
            tile_vae=args.tile_vae,
            tile_size=args.tile_size,
            overlap=args.overlap,
            model_dir=model_dir
        )
    
    # Get VAE info for clean printing
    vae_info = vae_system_instance.get_current_vae_info()
    print(f"Loading VAE: {vae_type} ({vae_info.get('description', '')})")

    # Initialize model manager and load DiT model
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    dit_path = f"{model_dir}/diffusion_pytorch_model_streaming_dmd.safetensors"
    
    # Load DiT model (Removed silence to catch errors)
    # print(f"Loading DiT model from {dit_path}...")
    try:
        with redirect_stdout(io.StringIO()):
            mm.load_models([dit_path])
    except Exception as error:
        raise RuntimeError(f"Cannot load DiT weights {dit_path}: {error}") from error
    
    if mm.fetch_model("wan_video_dit") is None:
        raise RuntimeError(f"DiT weights were not recognized: {dit_path}")

    # Create pipeline based on mode
    with redirect_stdout(io.StringIO()):
        if args.mode == "full":
            pipe = FlashVSRFullPipeline.from_model_manager(mm, device=args.device)
            pipe.vae = vae_model
        else:  # tiny or tiny-long
            if args.mode == "tiny":
                pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=args.device)
            else:  # tiny-long
                pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=args.device)
            pipe.TCDecoder = vae_model
    
    # Load and setup LQ projector efficiently
    lq_proj = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)
    lq_path = f"{model_dir}/LQ_proj_in.ckpt"
    load_checked_state_dict(lq_proj, read_checkpoint(lq_path), lq_path)
    lq_proj.eval()

    # Move to device and setup pipeline (combined operations)
    with redirect_stdout(io.StringIO()): # Suppress
        pipe.denoising_model().LQ_proj_in = lq_proj.to(args.device, dtype=dtype)
        pipe.to(args.device)
        pipe.enable_vram_management(num_persistent_param_in_dit=None)
        pipe.init_cross_kv()
        pipe.load_models_to_device(["dit", "vae"])
    
    return pipe, vae_system_instance

def run_inference(args, media):
    total_start_time = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats(args.device)

    # A path without a suffix is an output directory, even on a fresh checkout.
    if os.path.isdir(args.output) or not os.path.splitext(args.output)[1]:
        output_dir = args.output
    else:
        output_dir = os.path.dirname(os.path.abspath(args.output))
    # Always create output directory if it exists or not
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Mode: {args.mode}")
    # Prepare input
    print(f"Processing: {args.input}")
    
    # Set dtype
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    
    pipe, vae_instance = init_pipeline(args)
    LQ, th, tw, F, fps, input_video_path, total_frames_orig, exact_h, exact_w = prepare_input_tensor(
        args.input, 
        scale=args.scale, 
        dtype=dtype,
        device=args.device,
        media_info=args.input_info,
    )
    
    # Override FPS if specified by user
    if args.fps is not None:
        fps = args.fps
    
    
    # Ensure LQ is on the correct device (it should be already if prepare_input_tensor kept it there)
    # This is a no-op if already on device, but safe to keep.
    if LQ.device != torch.device(args.device):
         print(f"Moving input to {args.device}...")
         LQ = LQ.to(args.device)
    
    # Determine output file name
    input_name = Path(args.input).stem
    if os.path.isdir(args.output):
        # Generate unique output filename based on user provided parameters
        fn_parts = ["FlashVSR-Pro"]

        # Check if mode was explicitly provided in command line arguments
        mode_explicitly_set = any(arg.startswith("--mode") for arg in sys.argv)
        if mode_explicitly_set:
            fn_parts.append(args.mode)
        
        # Add scale
        fn_parts.append(f"scale{args.scale}")
        # Note: Seed is omitted as requested
        
        # Add optional parameter flags
        if args.keep_audio:
            fn_parts.append("audio")
        
        if args.fps is not None:
            fn_parts.append(f"fps{args.fps}")

        if args.color_fix:
            fn_parts.append("colorfix")
            
        if args.quality != 10:
            fn_parts.append(f"q{args.quality}")

        # Add advanced parameter flags if non-default
        if args.seed != 0:
            fn_parts.append(f"seed{args.seed}")
            
        if args.sparse_ratio != 2.0:
            fn_parts.append(f"sparse{args.sparse_ratio}")
            
        if args.kv_ratio != 3.0:
            fn_parts.append(f"kv{args.kv_ratio}")
            
        if args.local_range != 11:
            fn_parts.append(f"lr{args.local_range}")
            
        if args.dtype != "bf16":
            fn_parts.append(args.dtype)
            
        # Add tiling flags
        is_tiled = False
        if args.tile_dit:
            fn_parts.append("tile_dit")
            is_tiled = True
        
        if args.tile_vae:
            fn_parts.append("tile_vae")
            is_tiled = True
            
        # Append tile params only once if any tiling is enabled
        if is_tiled:
            # Append tile-size if non-default
            if args.tile_size != 256:
                fn_parts.append(f"ts{args.tile_size}")
            # Append overlap if non-default
            if args.overlap != 24:
                fn_parts.append(f"ol{args.overlap}")
            
        fn_parts.append(input_name)
        output_filename = "_".join(fn_parts) + ".mp4"
        output_path = os.path.join(args.output, output_filename)
    else:
        output_path = args.output
    
    # print(f"Output: {output_path}")
    
    # Prepare pipeline parameters
    pipeline_kwargs = {
        "prompt": "", 
        "negative_prompt": "", 
        "cfg_scale": 1.0, 
        "num_inference_steps": 1, 
        "seed": args.seed,
        "LQ_video": LQ, 
        "num_frames": F, 
        "height": th, 
        "width": tw, 
        "is_full_block": False, 
        "if_buffer": True,
        "topk_ratio": args.sparse_ratio * 768 * 1280 / (th * tw), 
        "kv_ratio": args.kv_ratio,
        "local_range": args.local_range,
        "color_fix": args.color_fix,
    }
    
    # Add VAE tiling parameters (for full and tiny mode)
    if args.tile_vae:
        pipeline_kwargs["tiled"] = True
        # Ensure we pass sensible tile_size and tile_stride for VAE
        # args.tile_size is from CLI (default 256 for simple Tile Utils, but here for VAE it is latent size?)
        # FlashVSRTiny default is (60, 104) ~= 480x832 pixels / 8
        # If user provides explicit tile size, use it. Otherwise, let's pick a reasonable default or respect args.tile_size
        
        # NOTE: args.tile_size is typically 256. 256 * 8 = 2048 pixels. This is a very large tile for VAE.
        # It's likely args.tile_size is meant for DiT pixel blocks in other contexts? 
        # But 'apply_tiled_inference_simple' uses it as pixel size for DiT.
        # For VAE here, 'tile_size' argument to __call__ expects Latent Size.
        # If we use 256 output pixels -> 32 latent size.
        # Let's assume if tile-dit is OFF, args.tile_size might be irrelevant or we should interpret it.
        # If tile-dit is ON, args.tile_size is used for DiT.
        
        # Let's interpret args.tile_size as PIXEL size for consistency if tile-dit is ON?
        # No, for VAE tiling, usually we want bigger chunks than DiT tiling. 
        # Let's use a safe default if not specified, or derive from args.tile_size / 8
        # If user didn't change default 256... 256/8 = 32. 
        
        # Actually, let's trust the defaults in the pipeline if user didn't specifying anything specific for VAE?
        # But user might want to control it using CLI args.
        # Let's pass args.tile_size // 8 if plausible.
        
        # Current logic: If tile-vae is ON, we want to enforce tiling.
        # Let's update tile_size and tile_stride in pipeline_kwargs
        
        # Safely convert pixel tile size (args.tile_size) to latent size
        # Assuming args.tile_size is "pixel size"
        vae_tile_size_latent = max(32, args.tile_size // 8)
        vae_overlap_latent = max(4, args.overlap // 8)
        
        pipeline_kwargs["tile_size"] = (vae_tile_size_latent, vae_tile_size_latent)
        pipeline_kwargs["tile_stride"] = (vae_tile_size_latent - vae_overlap_latent, vae_tile_size_latent - vae_overlap_latent)
        
        print(f"VAE Tiling Enabled: tile_size (latent)={pipeline_kwargs['tile_size']}, stride={pipeline_kwargs['tile_stride']}")

    validate_output(output_path, args.input, args.keep_audio)
    if args.metrics_json == Path(output_path).resolve():
        raise ValueError("Metrics must use a different path from video output")

    # Run inference (tiled or standard)
    torch.cuda.synchronize(args.device)
    inference_start_time = time.perf_counter()

    if args.tile_dit:
        print(f"Tiled DiT: tile_size={args.tile_size}, overlap={args.overlap}")

        # Create a copy of pipeline_kwargs and remove LQ_video
        tile_kwargs = pipeline_kwargs.copy()
        tile_kwargs.pop('LQ_video', None)  # Remove LQ_video because it's already passed as a positional argument

        # Handle potential collision of 'tile_size' argument
        # pipeline_kwargs might contain 'tile_size' (tuple) for VAE if tile-vae is on.
        # apply_tiled_inference_simple takes 'tile_size' (int) for DiT.
        # To avoid error, we pop 'tile_size' from kwargs and pass it as 'tile_size_vae' if present.
        vae_tile_size_tuple = tile_kwargs.pop('tile_size', None)
        
        # Tiled inference
        video = apply_tiled_inference_simple(
            pipe,
            LQ,
            tile_size=args.tile_size,
            overlap=args.overlap,
            tile_size_vae=vae_tile_size_tuple,
            **tile_kwargs
        )
    else:
        # print("Running inference...") # Controlled inside pipeline or tqdm
        if args.mode == 'tiny-long':
             msg = "Running inference (Streaming"
             if args.tile_vae:
                 msg += " & VAE-tiled"
             msg += ")..."
             print(msg)
        else:
             print("Running inference...")
        video = pipe(**pipeline_kwargs)

    torch.cuda.synchronize(args.device)
    inference_end_time = time.perf_counter()
    inference_duration = inference_end_time - inference_start_time
    print(f"Inference completed in {inference_duration:.2f} seconds")
    
    # Convert and save video
    # Crop back to exact requested resolution sH x sW (exact_h, exact_w)
    # The Model Output `video` is (B, C, T, H, W)
    if video.shape[-2] != exact_h or video.shape[-1] != exact_w:
        # We padded center, so we crop center
        curr_h, curr_w = video.shape[-2], video.shape[-1]
        pad_h = curr_h - exact_h
        pad_w = curr_w - exact_w
        
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        
        video = video[..., pad_top:pad_top+exact_h, pad_left:pad_left+exact_w]

    if video.shape[-2] < exact_h or video.shape[-1] < exact_w:
        raise RuntimeError("Pipeline returned smaller dimensions than requested")
    frames = tensor2video(video)
    if not frames:
        raise RuntimeError("Pipeline returned no frames")

    # Ensure output Duration matches Input Duration (Remove padding / Fill missing)
    if len(frames) > total_frames_orig:
        frames = frames[:total_frames_orig]
    elif len(frames) < total_frames_orig:
        # print(f"[Warning] Output frames ({len(frames)}) < Input ({total_frames_orig}). Padding to restore duration.")
        frames.extend([frames[-1]] * (total_frames_orig - len(frames)))

    output_info = media.save_video(
        frames, output_path, fps=fps, quality=args.quality,
        audio_source=input_video_path if args.keep_audio else None,
        device_index=torch.cuda.current_device(),
    )
    if args.metrics_json is not None:
        torch.cuda.synchronize(args.device)
        metrics = {
            "mode": args.mode, "device": torch.cuda.get_device_name(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
            "inference_seconds": inference_duration,
            "total_seconds": time.perf_counter() - total_start_time,
            "output": str(Path(output_path).resolve()), **output_info,
        }
        with atomic_output(args.metrics_json) as temporary:
            temporary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Done!\nOutput: {output_path}")

    # Process total time
    total_end_time = time.perf_counter()
    total_duration = total_end_time - total_start_time
    print(f"Total processing time: {total_duration:.2f} seconds")

    # Cleanup
    vae_instance.clean_memory()
    del pipe
    torch.cuda.empty_cache()

def _main(argv=None):
    args = parse_args(argv)
    args.input = str(validate_input(args.input))
    args.output = str(validate_output(args.output, args.input, args.keep_audio))
    if args.metrics_json is not None:
        args.metrics_json = args.metrics_json.expanduser().resolve()
        if args.metrics_json in (Path(args.input), Path(args.output)):
            raise ValueError("Metrics must use a different path from input and video output")
    validate_models(args.mode)
    media = MediaTools()
    args.input_info = media.verify_video(args.input) if Path(args.input).is_file() else None
    load_runtime(args, media)
    run_inference(args, media)
    return 0


def main(argv=None):
    return run_cli(_main, argv)


if __name__ == "__main__":
    raise SystemExit(main())
