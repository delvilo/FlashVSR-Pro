import logging
logger = logging.getLogger(__name__)

from typing import Optional

import torch
from flashvsr.progress import tqdm

from .color import TorchColorCorrectorWavelet
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline


class FlashVSRFullPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['dit', 'vae']
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.prompt_emb_posi = None
        self.ColorCorrector = TorchColorCorrectorWavelet(levels=5)

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        # Manage only dit / vae
        dtype = next(iter(self.dit.parameters())).dtype
        from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
        enable_vram_management(
            self.dit,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        self.enable_cpu_offload()


    def denoising_model(self):
        return self.dit

    # -------------------------
    # Added: Explicit KV pre-initialization function
    # -------------------------
    def init_cross_kv(
        self,
        context_tensor: Optional[torch.Tensor] = None,
    ):
        self.load_models_to_device(["dit"])
        """
        Use fixed prompt to generate text context and initialize KV cache for all CrossAttentions in WanModel.
        Must be called explicitly once before __call__.
        """
        if self.dit is None:
            raise RuntimeError("Initialize the FlashVSR DiT before initializing the prompt cache")

        if not isinstance(context_tensor, torch.Tensor) or context_tensor.numel() == 0:
            raise ValueError("init_cross_kv requires the verified fixed prompt tensor")
        ctx = context_tensor

        ctx = ctx.to(dtype=self.torch_dtype, device=self.device)

        if self.prompt_emb_posi is None:
            self.prompt_emb_posi = {}
        self.prompt_emb_posi['context'] = ctx

        if hasattr(self.dit, "reinit_cross_kv"):
            self.dit.reinit_cross_kv(ctx)
        else:
            raise AttributeError("WanModel missing reinit_cross_kv(ctx) method, please add it to implementation.")
        self.timestep = torch.tensor([1000.], device=self.device, dtype=self.torch_dtype)
        self.t = self.dit.time_embedding(sinusoidal_embedding_1d(self.dit.freq_dim, self.timestep))
        self.t_mod = self.dit.time_projection(self.t).unflatten(1, (6, self.dit.dim))
        # Scheduler
        self.scheduler.set_timesteps(1, denoising_strength=1.0, shift=5.0)
        self.load_models_to_device([])


    def prepare_extra_input(self, latents=None):
        return {}

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16), decoding_msg=None):
        # Always try to pass decoding_msg to the VAE
        # We assume the VAE (mostly WanVideoVAE) has been updated to accept this argument.
        
        if not decoding_msg:
             decoding_msg = "Decoding video"

        decode_kwargs = {
            "device": self.device,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "decoding_msg": decoding_msg
        }

        try:
            frames = self.vae.decode(latents, **decode_kwargs)
        except TypeError as e:
            # Fallback for VAEs that don't accept decoding_msg
            if "unexpected keyword argument 'decoding_msg'" in str(e):
                decode_kwargs.pop("decoding_msg")
                # Print manually since VAE won't handle the description
                if decoding_msg:
                    logger.info(decoding_msg)
                else:
                    logger.info("Decoding video...")
                frames = self.vae.decode(latents, **decode_kwargs)
            else:
                raise e
                
        return frames

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        negative_prompt="",
        denoising_strength=1.0,
        seed=None,
        rand_device="gpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(60, 104),
        tile_stride=(30, 52),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="Wan2.1-T2V-1.3B",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        LQ_video=None,
        is_full_block=False,
        if_buffer=False,
        topk_ratio=2.0,
        kv_ratio=3.0,
        local_range = 9,
        color_fix = True,
        decoding_msg = None,
    ):
        # Only accept cfg=1.0 (Consistent with original code)
        assert cfg_scale == 1.0, "cfg_scale must be 1.0"

        # Requirement: init_cross_kv() must be called first
        if self.prompt_emb_posi is None or 'context' not in self.prompt_emb_posi:
            raise RuntimeError(
                "Cross-Attn KV not initialized. Please call before __call__:\n"
                "    pipe.init_cross_kv()\n"
                "Or pass custom context:\n"
                "    pipe.init_cross_kv(context_tensor=your_context_tensor)"
            )

        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            logger.info(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")

        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Initialize noise
        if if_buffer:
            noise = self.generate_noise((1, 16, (num_frames - 1) // 4, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)
        else:
            noise = self.generate_noise((1, 16, (num_frames - 1) // 4 + 1, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)
        # noise = noise.to(dtype=self.torch_dtype, device=self.device)
        latents = noise

        process_total_num = (num_frames - 1) // 8 - 2
        is_stream = True

        # Clear potential LQ_proj_in cache
        if hasattr(self.dit, "LQ_proj_in"):
            self.dit.LQ_proj_in.clear_cache()


        latents_total = []
        self.vae.clear_cache()

        with torch.no_grad():
            pbar = tqdm(range(process_total_num), desc="DiT Inference")
            for cur_process_idx in pbar:
                # Removed synchronization for speed
                # torch.cuda.synchronize()
                # dit_time_start = time.time()

                if cur_process_idx == 0:
                    pre_cache_k = [None] * len(self.dit.blocks)
                    pre_cache_v = [None] * len(self.dit.blocks)
                    LQ_latents = None
                    inner_loop_num = 7
                    for inner_idx in range(inner_loop_num):
                        cur = self.denoising_model().LQ_proj_in.stream_forward(
                            LQ_video[:, :, max(0, inner_idx*4-3):(inner_idx+1)*4-3, :, :]
                        ) if LQ_video is not None else None
                        if cur is None:
                            continue
                        if LQ_latents is None:
                            LQ_latents = cur
                        else:
                            for layer_idx in range(len(LQ_latents)):
                                LQ_latents[layer_idx] = torch.cat([LQ_latents[layer_idx], cur[layer_idx]], dim=1)
                    cur_latents = latents[:, :, :6, :, :]
                else:
                    LQ_latents = None
                    inner_loop_num = 2
                    for inner_idx in range(inner_loop_num):
                        cur = self.denoising_model().LQ_proj_in.stream_forward(
                            LQ_video[:, :, cur_process_idx*8+17+inner_idx*4:cur_process_idx*8+21+inner_idx*4, :, :]
                        ) if LQ_video is not None else None
                        if cur is None:
                            continue
                        if LQ_latents is None:
                            LQ_latents = cur
                        else:
                            for layer_idx in range(len(LQ_latents)):
                                LQ_latents[layer_idx] = torch.cat([LQ_latents[layer_idx], cur[layer_idx]], dim=1)
                    cur_latents = latents[:, :, 4+cur_process_idx*2:6+cur_process_idx*2, :, :]

                # Inference (No motion_controller / vace)
                noise_pred_posi, pre_cache_k, pre_cache_v = model_fn_wan_video(
                    self.dit,
                    x=cur_latents,
                    timestep=self.timestep,
                    context=None,
                    tea_cache=None,
                    LQ_latents=LQ_latents,
                    is_full_block=is_full_block,
                    is_stream=is_stream,
                    pre_cache_k=pre_cache_k,
                    pre_cache_v=pre_cache_v,
                    topk_ratio=topk_ratio,
                    kv_ratio=kv_ratio,
                    cur_process_idx=cur_process_idx,
                    t_mod=self.t_mod,
                    t=self.t,
                    local_range = local_range,
                )

                # Update latent
                cur_latents = cur_latents - noise_pred_posi
                latents_total.append(cur_latents)

            latents = torch.cat(latents_total, dim=2)

            # Decode
            tiler_kwargs["decoding_msg"] = decoding_msg
            frames = self.decode_video(latents, **tiler_kwargs)

            # Color correction (wavelet)
            try:
                if color_fix:
                    frames = self.ColorCorrector(
                        frames.to(device=LQ_video.device),
                        LQ_video[:, :, :frames.shape[2], :, :],
                        clip_range=(-1, 1),
                        chunk_size=16,
                        method='adain'
                    )
            except Exception as e:
                raise RuntimeError(f"Color correction failed: {e}") from e

        return frames[0]


# -----------------------------
# Simplified Model Forward Wrapper (No vace / No motion_controller)
# -----------------------------
def model_fn_wan_video(
    dit: WanModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    tea_cache: Optional[object] = None, # Reserved for compatibility
    LQ_latents: Optional[torch.Tensor] = None,
    is_full_block: bool = False,
    is_stream: bool = False,
    pre_cache_k: Optional[list[torch.Tensor]] = None,
    pre_cache_v: Optional[list[torch.Tensor]] = None,
    topk_ratio: float = 2.0,
    kv_ratio: float = 3.0,
    cur_process_idx: int = 0,
    t_mod : torch.Tensor = None,
    t : torch.Tensor = None,
    local_range: int = 9,
    **kwargs,
):
    # patchify
    x, (f, h, w) = dit.patchify(x)

    win = (2, 8, 8)
    seqlen = f // win[0]
    local_num = seqlen
    window_size = win[0] * h * w // 128
    square_num = window_size * window_size
    topk = int(square_num * topk_ratio) - 1
    kv_len = int(kv_ratio)

    # RoPE Position (Segmented)
    if cur_process_idx == 0:
        freqs = torch.cat([
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    else:
        freqs = torch.cat([
            dit.freqs[0][4 + cur_process_idx*2:4 + cur_process_idx*2 + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    # Unified Sequence Parallel (Default OFF)

    # Block Stacking
    for block_id, block in enumerate(dit.blocks):
        if LQ_latents is not None and block_id < len(LQ_latents):
            x = x + LQ_latents[block_id]
        x, last_pre_cache_k, last_pre_cache_v = block(
            x, context, t_mod, freqs, f, h, w,
            local_num, topk,
            block_id=block_id,
            kv_len=kv_len,
            is_full_block=is_full_block,
            is_stream=is_stream,
            pre_cache_k=pre_cache_k[block_id] if pre_cache_k is not None else None,
            pre_cache_v=pre_cache_v[block_id] if pre_cache_v is not None else None,
            local_range = local_range,
        )
        if pre_cache_k is not None: pre_cache_k[block_id] = last_pre_cache_k
        if pre_cache_v is not None: pre_cache_v[block_id] = last_pre_cache_v

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x, pre_cache_k, pre_cache_v
