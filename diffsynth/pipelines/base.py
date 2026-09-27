import logging
logger = logging.getLogger(__name__)

import torch


class BasePipeline(torch.nn.Module):
    def __init__(self, device="cuda", torch_dtype=torch.float16, height_division_factor=64, width_division_factor=64):
        super().__init__()
        self.device = device
        self.torch_dtype = torch_dtype
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.cpu_offload = False
        self.model_names = []

    def check_resize_height_width(self, height, width):
        if height % self.height_division_factor != 0:
            height = (height + self.height_division_factor - 1) // self.height_division_factor * self.height_division_factor
            logger.info(f"The height cannot be evenly divided by {self.height_division_factor}. We round it up to {height}.")
        if width % self.width_division_factor != 0:
            width = (width + self.width_division_factor - 1) // self.width_division_factor * self.width_division_factor
            logger.info(f"The width cannot be evenly divided by {self.width_division_factor}. We round it up to {width}.")
        return height, width

    def enable_cpu_offload(self):
        self.cpu_offload = True

    def load_models_to_device(self, loadmodel_names=()):
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            return
        # offload the unneeded models to cpu
        for model_name in self.model_names:
            if model_name not in loadmodel_names:
                model = getattr(self, model_name)
                if model is not None:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                module.offload()
                    else:
                        model.cpu()
        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if model is not None:
                if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            module.onload()
                else:
                    model.to(self.device)
        # fresh the cuda cache
        torch.cuda.empty_cache()

    def generate_noise(self, shape, seed=None, device="cpu", dtype=torch.float16):
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise
