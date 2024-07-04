from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from public import public


@public
@partial(jax.jit, inline=True)
def identity(d: ArrayLike) -> Array:
    """
    Identity function that returns the input distance.

    Examples:
        >>> identity(1.0)
        Array(1., dtype=...)
    """
    return jnp.asarray(d)
