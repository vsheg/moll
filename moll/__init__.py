"""
This module serves as the initialization file for the `moll` package.
"""

__version__ = "0.1.11"  # TODO: automatically update this

# Determine the current execution mode
import sys as _sys

DEBUG = _sys.gettrace() is not None
TEST = "pytest" in _sys.modules
JIT = not DEBUG

# Uncomment the line below to manually disable just-in-time compilation
# JIT = False

if not JIT:
    # Disable just-in-time compilation
    import jax as _jax

    _jax.config.update("jax_disable_jit", True)

if TEST:
    # Disable memory preallocation for tests as they are run in parallel
    import os as _os

    _os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    _os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "gpu"

# Submodules are available as attributes
from . import (  # noqa: F401 E402 (unused import, import at top of file)
    metrics,
    pick,
    typing,
    utils,
)
