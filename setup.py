"""Compatibility shim; package metadata and dependencies live in pyproject.toml."""

import sys

if not sys.platform.startswith("linux"):
    raise RuntimeError("FlashVSR-Pro supports Linux and Google Colab only.")

from setuptools import setup

setup()
