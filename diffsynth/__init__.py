"""Minimal FlashVSR-only DiffSynth subset; public pipelines are loaded lazily."""
from importlib import import_module

_EXPORTS = {
    "FlashVSRFullPipeline": "flashvsr_full",
    "FlashVSRTinyPipeline": "flashvsr_tiny",
    "FlashVSRTinyLongPipeline": "flashvsr_tiny_long",
}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f".pipelines.{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value
