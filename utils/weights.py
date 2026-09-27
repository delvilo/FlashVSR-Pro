"""Checkpoint loading that never continues with missing or incompatible weights."""

from collections.abc import Mapping
from pathlib import Path

from .runtime import require_weight


def read_checkpoint(path):
    path = require_weight(path)
    try:
        if path.suffix == '.safetensors':
            from safetensors.torch import load_file
            checkpoint = load_file(str(path), device='cpu')
        else:
            import torch
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if isinstance(checkpoint, Mapping) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        if not isinstance(checkpoint, Mapping) or not checkpoint:
            raise ValueError('checkpoint does not contain a nonempty state dictionary')
        return checkpoint
    except Exception as error:
        raise RuntimeError(f'Cannot load model weights {path}: {error}') from error


def load_checked_state_dict(model, state_dict, source, allowed_missing=()):
    try:
        result = model.load_state_dict(state_dict, strict=False)
        missing = [key for key in result.missing_keys if not key.startswith(allowed_missing)]
        if missing or result.unexpected_keys:
            raise ValueError(
                f'missing keys: {missing[:5]}; unexpected keys: {result.unexpected_keys[:5]}'
            )
    except Exception as error:
        raise RuntimeError(f'Incompatible model weights {source}: {error}') from error
