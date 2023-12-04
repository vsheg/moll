from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax


def _pairwise_distances(X, dist_fn: Callable):
    """Compute pairwise distances between points in X using a custom distance function."""

    def x_to_X_dists(x):
        return jax.vmap(lambda y: dist_fn(x, y))(X)

    dists = jax.vmap(x_to_X_dists)(X)

    return dists


def _matrix_cross_sum(X: jnp.ndarray, i: int, j: int, row_only=False):
    """
    Computes the sum of the elements in the row `i` and the column `j` of the matrix `X`.
    """

    return lax.cond(
        row_only,
        lambda X: X[i, :].sum(),
        lambda X: X[i, :].sum() + X[:, j].sum() - X[i, j],
        X,
    )
