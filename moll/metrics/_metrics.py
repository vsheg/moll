"""
This module contains implementations of various metrics for comparing vectors.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.numpy import array as A  # noqa: F401 (unused import), used in doctests
from public import public


@public
@partial(jax.jit, inline=True)
def mismatches(p1: Array, p2: Array):
    """
    Computes the L0-norm distance between two vectors.

    Examples:
        >>> mismatches([1, 2, 3], [1, 2, 3])
        Array(0., dtype=float32)

        >>> mismatches([1, 2, 3], [1, 2, -3])
        Array(1., dtype=float32)

        >>> mismatches([1, 2, 3], [4, 5, 6])
        Array(3., dtype=float32)
    """
    p1 = jnp.asarray(p1)
    p2 = jnp.asarray(p2)
    return jnp.sum(p1 != p2).astype(float)


@public
@partial(jax.jit, inline=True)
def manhattan(p1: Array, p2: Array):
    """
    Computes the Manhattan distance between two vectors.

    Examples:
        >>> manhattan([1, 2, 3], [1, 2, 3])
        Array(0., dtype=float32)

        >>> manhattan([1, 2, 3], [1, 2, 4])
        Array(1., dtype=float32)

        >>> manhattan([1, 2, 3], [4, 5, 6])
        Array(9., dtype=float32)
    """
    p1 = jnp.asarray(p1)
    p2 = jnp.asarray(p2)
    return jnp.sum(jnp.abs(p1 - p2)).astype(float)


@public
@partial(jax.jit, inline=True)
def euclidean(p1: Array, p2: Array):
    """
    Computes the Euclidean distance between two vectors.

    Examples:
        >>> euclidean([1, 2, 3], [1, 2, 3])
        Array(0., dtype=float32)

        >>> euclidean([1, 2, 3], [1, 2, 4])
        Array(1., dtype=float32)

        >>> euclidean([1, 2, 3], [4, 5, 6])
        Array(5.196152, dtype=float32)
    """
    p1 = jnp.asarray(p1)
    p2 = jnp.asarray(p2)
    return jnp.linalg.norm(p1 - p2)


@public
@partial(jax.jit, inline=True)
def cosine(a: Array, b: Array):
    """
    Computes the cosine distance between two vectors.

    Examples:
        >>> cosine([1, 0], [1, 0])
        Array(1., dtype=float32)

        >>> cosine([1, 0], [0, 1])
        Array(0., dtype=float32)

        >>> cosine([1, 0], [-1, 0])
        Array(-1., dtype=float32)
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))


@public
@partial(jax.jit, inline=True)
def negative_cosine(a: Array, b: Array):
    """
    Computes the cosine distance between two vectors.

    Examples:
        >>> negative_cosine([1, 0], [1, 0])
        Array(-1., dtype=float32)

        >>> negative_cosine([1, 0], [0, 1])
        Array(-0., dtype=float32)

        >>> negative_cosine([1, 0], [-1, 0])
        Array(1., dtype=float32)
    """
    return -cosine(a, b)


@public
@partial(jax.jit, inline=True)
def tanimoto(a: Array, b: Array):
    """
    Computes the Tanimoto coefficient between two vectors.

    Examples:
        >>> tanimoto([1, 1], [1, 0])
        Array(0.5, dtype=float32)

        >>> tanimoto([1, 1], [0, 0])
        Array(0., dtype=float32)
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)

    bitwise_or = jnp.bitwise_or(a, b).sum().astype(float)
    bitwise_and = jnp.bitwise_and(a, b).sum().astype(float)

    # Check for the case where both vectors are all zeros and return 0.0 in that case

    return lax.cond(
        bitwise_or == 0.0,
        lambda: 0.0,
        lambda: bitwise_and / bitwise_or,
    )


@public
@partial(jax.jit, inline=True)
def one_minus_tanimoto(a: Array, b: Array):
    """
    Computes the Tanimoto distance between two vectors.

    Examples:
        >>> one_minus_tanimoto([1, 1], [1, 0])
        Array(0.5, dtype=float32)

        >>> one_minus_tanimoto([1, 1], [0, 0])
        Array(1., dtype=float32)
    """
    return 1.0 - tanimoto(a, b)
