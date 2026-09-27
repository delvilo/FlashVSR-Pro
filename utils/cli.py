"""Compatibility imports; application configuration lives in flashvsr."""
from flashvsr.cli import positive_float, run_cli, segment_seconds
from flashvsr.cli import parse_args as _parse_args
from flashvsr.config import IMAGE_SUFFIXES, MODES, OUTPUT_SUFFIXES, VIDEO_SUFFIXES, validate_input, validate_output
import sys


def parse_args(argv=None):
    return _parse_args(["infer", *(sys.argv[1:] if argv is None else argv)])
