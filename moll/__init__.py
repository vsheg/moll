import sys

import jax

DEBUG = sys.gettrace() is not None
TEST = "pytest" in sys.modules
JIT = not DEBUG

# JIT = False

# TODO: not working with pytest-cov
if not JIT:
    jax.config.update("jax_disable_jit", True)
