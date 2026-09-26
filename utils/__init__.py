"""FlashVSR utilities; lightweight checks can be used before installing PyTorch."""

from importlib import import_module

_EXPORTS = {
    "RMS_norm": "utils",
    "CausalConv3d": "utils",
    "PixelShuffle3d": "utils",
    "Buffer_LQ4x_Proj": "utils",
    "Causal_LQ4x_Proj": "utils",
    "build_tcdecoder": "TCDecoder",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
