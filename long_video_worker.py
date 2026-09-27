#!/usr/bin/env python3
"""Compatibility entry point for ``flashvsr long``."""
from flashvsr.cli import legacy_main


def main(argv=None):
    return legacy_main("long", argv)


if __name__ == "__main__":
    raise SystemExit(main())
