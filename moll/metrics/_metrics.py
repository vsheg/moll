"""
This module contains implementations of various metrics for comparing vectors.
"""

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.numpy import array as A  # noqa: F401 (unused import), used in doctests
from public import public


@public
@jax.jit
def mismatches(p1: Array, p2: Array):
    """
    Computes the L0-norm distance between two vectors.

    >>> mismatches(A([1, 2, 3]), A([1, 2, 3])).item()
    0.0

    >>> mismatches(A([1, 2, 3]), A([1, 2, -3])).item()
    1.0

    >>> mismatches(A([1, 2, 3]), A([4, 5, 6])).item()
    3.0
    """
    return jnp.sum(p1 != p2).astype(float)


@public
@jax.jit
def manhattan(p1: Array, p2: Array):
    """
    Computes the Manhattan distance between two vectors.

    >>> manhattan(A([1, 2, 3]), A([1, 2, 3])).item()
    0.0

    >>> manhattan(A([1, 2, 3]), A([1, 2, 4])).item()
    1.0

    >>> manhattan(A([1, 2, 3]), A([4, 5, 6])).item()
    9.0
    """
    return jnp.sum(jnp.abs(p1 - p2)).astype(float)


@public
@jax.jit
def euclidean(p1: Array, p2: Array):
    """
    Computes the Euclidean distance between two vectors.

    >>> euclidean(A([1, 2, 3]), A([1, 2, 3])).item()
    0.0

    >>> euclidean(A([1, 2, 3]), A([1, 2, 4])).item()
    1.0

    >>> euclidean(A([1, 2, 3]), A([4, 5, 6])).item()
    5.19615...
    """
    return jnp.linalg.norm(p1 - p2)


@public
@jax.jit
def cosine(a: Array, b: Array):
    """
    Computes the cosine distance between two vectors.

    >>> cosine(A([1, 0]), A([1, 0])).item()
    1.0

    >>> cosine(A([1, 0]), A([0, 1])).item()
    0.0

    >>> cosine(A([1, 0]), A([-1, 0])).item()
    -1.0
    """
    return (jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b))).astype(float)


@public
def negative_cosine(a: Array, b: Array):
    """
    Computes the cosine distance between two vectors.
    """
    return -cosine(a, b)


@public
@jax.jit
def tanimoto(a: Array, b: Array):
    """
    Computes the Tanimoto coefficient between two vectors.

    >>> tanimoto(A([1, 1]), A([1, 0])).item()
    0.5

    >>> tanimoto(A([1, 1]), A([0, 0])).item()
    0.0
    """
    bitwise_or = jnp.bitwise_or(a, b).sum().astype(float)
    bitwise_and = jnp.bitwise_and(a, b).sum().astype(float)

    # Check for the case where both vectors are all zeros and return 0.0 in that case

    return lax.cond(
        bitwise_or == 0.0,
        lambda: 0.0,
        lambda: bitwise_and / bitwise_or,
    )


@public
@jax.jit
def one_minus_tanimoto(a: Array, b: Array):
    """
    Computes the Tanimoto distance between two vectors.
    """
    return 1.0 - tanimoto(a, b)