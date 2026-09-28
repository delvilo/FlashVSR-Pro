"""Strict loading of the single FlashVSR architecture and its selected decoder."""

from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


def reset_pipeline(pipe):
    """Discard video-dependent state while retaining weights and fixed-prompt KV."""
    if pipe.dit is not None:
        projector = getattr(pipe.dit, "LQ_proj_in", None)
        if projector is not None:
            projector.clear_cache()
        for block in pipe.dit.blocks:
            block.self_attn.local_attn_mask = None
    decoder = getattr(pipe, "TCDecoder", None)
    if decoder is not None:
        decoder.clean_mem()
    vae = getattr(pipe, "vae", None)
    if vae is not None:
        vae.clear_cache()


@contextmanager
def load_pipeline(config, registry):
    # Heavy imports happen only after the engine has verified files and CUDA.
    import torch
    from diffsynth.models.wan_video_dit import WanModel
    from diffsynth.models.utils import hash_state_dict_keys, init_weights_on_device
    from utils.utils import Causal_LQ4x_Proj
    from utils.vae_manager import VAESystem
    from utils.weights import load_checked_state_dict, read_checkpoint

    if config.mode == "full":
        from diffsynth.pipelines.flashvsr_full import FlashVSRFullPipeline as Pipeline
    elif config.mode == "tiny":
        from diffsynth.pipelines.flashvsr_tiny import FlashVSRTinyPipeline as Pipeline
    else:
        from diffsynth.pipelines.flashvsr_tiny_long import FlashVSRTinyLongPipeline as Pipeline

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[config.dtype]
    vae = VAESystem(device=config.device, dtype=dtype)
    pipe = None
    try:
        logger.info("Loading FlashVSR %s (%s) from %s", registry.version, config.mode, registry.directory)
        state = read_checkpoint(registry.directory / "diffusion_pytorch_model_streaming_dmd.safetensors")
        if hash_state_dict_keys(state) != registry.release["state_dict_shape_hash"]:
            raise ValueError("DiT tensor keys/shapes do not match the FlashVSR model manifest")
        with init_weights_on_device():
            dit = WanModel(**registry.release["architecture"])
        dit.load_state_dict(state, strict=True, assign=True)
        del state
        dit = dit.eval().to(dtype=dtype)
        pipe = Pipeline(device=config.device, torch_dtype=dtype)
        pipe.dit = dit
        decoder = vae.load_vae(
            vae_type="wan2.1" if config.mode == "full" else "tcd", mode=config.mode,
            model_dir=str(registry.directory), tile_vae=config.tile_vae,
            tile_size=config.tile_size, overlap=config.overlap,
        )
        if config.mode == "full":
            pipe.vae = decoder
        else:
            pipe.TCDecoder = decoder
        projector = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1)
        path = registry.directory / "LQ_proj_in.ckpt"
        load_checked_state_dict(projector, read_checkpoint(path), path)
        pipe.dit.LQ_proj_in = projector.eval().to(config.device, dtype=dtype)
        pipe.to(config.device)
        pipe.enable_vram_management(num_persistent_param_in_dit=None)
        prompt = torch.load(registry.directory / "posi_prompt.pth", map_location="cpu", weights_only=True)
        if not isinstance(prompt, torch.Tensor) or not torch.isfinite(prompt).all():
            raise ValueError("Invalid fixed prompt tensor")
        pipe.init_cross_kv(context_tensor=prompt)
        pipe.load_models_to_device(["dit", "vae"])
        yield pipe
    finally:
        # The engine releases tensor references before emptying the CUDA cache.
        vae.clean_memory()
        if pipe is not None and pipe.dit is not None:
            pipe.dit.clear_cross_kv()
