"""
This module contains implementations of various metrics for comparing vectors.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.typing import ArrayLike
from public import public


@public
@partial(jax.jit, inline=True)
def mismatches(u: ArrayLike, v: ArrayLike) -> Array:
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
    u = jnp.asarray(u)
    v = jnp.asarray(v)
    return jnp.sum(u != v).astype(float)


@public
@partial(jax.jit, inline=True)
def manhattan(u: ArrayLike, v: ArrayLike) -> Array:
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
    u = jnp.asarray(u)
    v = jnp.asarray(v)
    return jnp.sum(jnp.abs(u - v)).astype(float)


@public
@partial(jax.jit, inline=True)
def euclidean(u: ArrayLike, v: ArrayLike) -> Array:
    """
    Computes the Euclidean distance (L2-norm) $||x - y||_2$ between two vectors.

    Examples:
        >>> euclidean([1, 2, 3], [1, 2, 3])
        Array(0., dtype=float32)

        >>> euclidean([1, 2, 3], [1, 2, 4])
        Array(1., dtype=float32)

        >>> euclidean([1, 2, 3], [4, 5, 6])
        Array(5.196152, dtype=float32)
    """
    u = jnp.asarray(u)
    v = jnp.asarray(v)
    return jnp.linalg.norm(u - v)


@public
@partial(jax.jit, inline=True)
def cosine(u: ArrayLike, v: ArrayLike) -> Array:
    r"""
    Computes the cosine similarity between two vectors:
    $$ \cos \widehat{\bf u, \bf v} = \dfrac{\bf u \cdot \bf v}{\norm{\bf u} \cdot \norm{\bf v}}, $$
    this formula follows from the dot product expression:
    $$ \bf u \cdot \bf v = \norm{\bf u} \cdot \norm{\bf v} \cdot \cos \widehat{\bf u, \bf v}. $$

    Examples:
        >>> cosine([1, 0], [1, 0])
        Array(1., dtype=float32)

        >>> cosine([1, 0], [0, 1])
        Array(0., dtype=float32)

        >>> cosine([1, 0], [-1, 0])
        Array(-1., dtype=float32)
    """
    u = jnp.asarray(u)
    v = jnp.asarray(v)
    return jnp.dot(u, v) / (jnp.linalg.norm(u) * jnp.linalg.norm(v))


@public
@partial(jax.jit, inline=True)
def negative_cosine(u: ArrayLike, v: ArrayLike) -> Array:
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
    return -cosine(u, v)


@public
@partial(jax.jit, inline=True)
def tanimoto(u: ArrayLike, v: ArrayLike) -> Array:
    """
    Computes the Tanimoto coefficient between two vectors.

    Examples:
        >>> tanimoto([1, 1], [1, 0])
        Array(0.5, dtype=float32)

        >>> tanimoto([1, 1], [0, 0])
        Array(0., dtype=float32)
    """
    u = jnp.asarray(u)
    v = jnp.asarray(v)

    bitwise_or = jnp.bitwise_or(u, v).sum().astype(float)
    bitwise_and = jnp.bitwise_and(u, v).sum().astype(float)

    # Check for the case where both vectors are all zeros and return 0.0 in that case

    return lax.cond(
        bitwise_or == 0.0,
        lambda: 0.0,
        lambda: bitwise_and / bitwise_or,
    )


@public
@partial(jax.jit, inline=True)
def one_minus_tanimoto(u: ArrayLike, v: ArrayLike) -> Array:
    """
    Computes the Tanimoto distance between two vectors.

    Examples:
        >>> one_minus_tanimoto([1, 1], [1, 0])
        Array(0.5, dtype=float32)

        >>> one_minus_tanimoto([1, 1], [0, 0])
        Array(1., dtype=float32)
    """
    return 1.0 - tanimoto(u, v)
