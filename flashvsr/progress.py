"""Interactive progress follows the application's log level."""

import logging
import sys
from tqdm import tqdm as _tqdm


def tqdm(*args, **kwargs):
    if not sys.stderr.isatty() or not logging.getLogger().isEnabledFor(logging.INFO):
        kwargs["disable"] = True
    return _tqdm(*args, **kwargs)
