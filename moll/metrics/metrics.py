import jax
import jax.numpy as jnp

__all__ = [
    "mismatches",
    "manhattan",
    "euclidean",
    "tanimoto",
    "one_minus_tanimoto",
]


@jax.jit
def mismatches(p1, p2):
    """
    Computes the L0-norm distance between two vectors.

    >>> mismatches(jnp.array([1, 2, 3]), jnp.array([1, 2, 3])).item()
    0.0

    >>> mismatches(jnp.array([1, 2, 3]), jnp.array([1, 2, -3])).item()
    1.0

    >>> mismatches(jnp.array([1, 2, 3]), jnp.array([4, 5, 6])).item()
    3.0
    """
    return jnp.sum(p1 != p2).astype(float)


@jax.jit
def manhattan(p1, p2):
    """
    Computes the Manhattan distance between two vectors.

    >>> manhattan(jnp.array([1, 2, 3]), jnp.array([1, 2, 3])).item()
    0.0

    >>> manhattan(jnp.array([1, 2, 3]), jnp.array([1, 2, 4])).item()
    1.0

    >>> manhattan(jnp.array([1, 2, 3]), jnp.array([4, 5, 6])).item()
    9.0
    """
    return jnp.sum(jnp.abs(p1 - p2)).astype(float)


@jax.jit
def euclidean(p1, p2):
    """
    Computes the Euclidean distance between two vectors.

    >>> euclidean(jnp.array([1, 2, 3]), jnp.array([1, 2, 3])).item()
    0.0

    >>> euclidean(jnp.array([1, 2, 3]), jnp.array([1, 2, 4])).item()
    1.0

    >>> euclidean(jnp.array([1, 2, 3]), jnp.array([4, 5, 6])).item()
    5.19615...
    """
    return jnp.linalg.norm(p1 - p2)


@jax.jit
def tanimoto(a: jnp.ndarray, b: jnp.ndarray):
    """
    Computes the Tanimoto coefficient between two vectors.

    >>> tanimoto(jnp.array([1, 1]), jnp.array([1, 0])).item()
    0.5

    >>> tanimoto(jnp.array([1, 1]), jnp.array([0, 0])).item()
    0.0
    """

    bitwise_or = jnp.bitwise_or(a, b).sum().astype(float)
    bitwise_and = jnp.bitwise_and(a, b).sum().astype(float)

    # Check for the case where both vectors are all zeros and return 0.0 in that case

    return jax.lax.cond(
        bitwise_or == 0.0,
        lambda: 0.0,
        lambda: bitwise_and / bitwise_or,
    )


@jax.jit
def one_minus_tanimoto(a: jnp.ndarray, b: jnp.ndarray):
    """
    Computes the Tanimoto distance between two vectors.
    """

    return 1.0 - tanimoto(a, b)
