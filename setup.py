"""Compatibility shim; package metadata and dependencies live in pyproject.toml."""

import sys

if not (3, 12) <= sys.version_info[:2] < (3, 15):
    raise RuntimeError("FlashVSR-Pro requires Python 3.12–3.14.")

if not sys.platform.startswith("linux"):
    raise RuntimeError("FlashVSR-Pro supports Linux and Google Colab only.")

from setuptools import setup

setup()
