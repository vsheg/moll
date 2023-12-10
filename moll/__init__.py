import sys as _sys

from . import metrics, picking, typing, utils  # noqa: F401

DEBUG = _sys.gettrace() is not None
TEST = "pytest" in _sys.modules
JIT = not DEBUG

# JIT = False

# TODO: not working with pytest-cov
if not JIT:
    import jax

    jax.config.update("jax_disable_jit", True)

if TEST:
    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "gpu"
