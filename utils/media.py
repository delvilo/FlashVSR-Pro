"""Compatibility imports; video and audio I/O live in flashvsr.media."""
from flashvsr.media import MediaError, MediaTools, atomic_output

__all__ = ["MediaError", "MediaTools", "atomic_output"]
