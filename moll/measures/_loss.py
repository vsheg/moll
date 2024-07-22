from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from public import public


@public
@partial(jax.jit, inline=True)
def power(diff: ArrayLike, p: float = 2.0) -> Array:
    """
    Computes the power loss for a given difference.

    Examples:
        >>> dist = 10.0

        >>> power(dist)
        Array(100., dtype=...)

        >>> power(dist, 3.0)
        Array(1000., dtype=...)
    """
    diff = jnp.asarray(diff)

    return jnp.where(
        jnp.isclose(diff, 0),
        jnp.inf,
        jnp.power(diff, p),
    )


@public
@partial(jax.jit, inline=True)
def exponential(diff: ArrayLike, p: float = 1.0) -> Array:
    """
    Computes the exponential loss for a given error.

    Examples:
        >>> dist = 10.0

        >>> exponential(dist)
        Array(22026.465, dtype=...)

        >>> exponential(dist, 2.0)
        Array(4.851652e+08, dtype=...)
    """
    diff = jnp.asarray(diff)
    return jnp.exp(p * diff)


@public
@partial(jax.jit, inline=True)
def lennard_jones(diff: ArrayLike, p: float = 1.0) -> Array:
    """
    Computes the Lennard-Jones potential (loss) for a given distance.

    Examples:
        >>> lennard_jones(10.0)
        Array(-4.999997e-07, dtype=...)

        >>> lennard_jones(0.01)
        Array(2.5000008e+23, dtype=...)
    """
    diff = jnp.asarray(diff)
    sigma: float = p * 0.5 ** (1 / 6)
    return jnp.where(
        diff > 0,
        jnp.power(sigma / diff, 12.0) - jnp.power(sigma / diff, 6.0),
        jnp.inf,
    )


@public
@partial(jax.jit, inline=True)
def logarithmic(
    diff: ArrayLike,
    p: float = 1.0,
    negative_diff_result: float = -jnp.inf,
) -> Array:
    """
    Computes the logarithmic loss for a given distance.

    Examples:
        >>> dist = 10.0

        >>> logarithmic(dist)
        Array(2.3025851, dtype=...)

        Negative `p` is considered by its absolute value:
        >>> logarithmic(dist, p=-2)
        Array(2.9957323, dtype=...)
    """
    diff = jnp.asarray(diff)
    return jnp.where(
        diff > 0,
        jnp.log(jnp.abs(p) * diff),
        negative_diff_result,
    )
