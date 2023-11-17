import sys
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax
from loguru import logger

from moll.core.utils import fill_diagonal

DEBUG = sys.gettrace() is not None
TEST = "pytest" in sys.modules
JIT = not DEBUG

# JIT = False

# TODO: not working with pytest-cov
if not JIT:
    jax.config.update("jax_disable_jit", True)


@jax.jit
def tanimoto(a: jnp.ndarray, b: jnp.ndarray) -> float:
    """
    Computes the Tanimoto distance between two vectors.
    """

    bitwise_or = jnp.bitwise_or(a, b).sum().astype(float)
    bitwise_and = jnp.bitwise_and(a, b).sum().astype(float)

    # Check for the case where both vectors are all zeros and return 0.0 in that case
    dist = jax.lax.cond(
        bitwise_or == 0.0,
        lambda _: 0.0,
        lambda _: 1 - (bitwise_and / bitwise_or),
        None,
    )

    return dist


@jax.jit
def euclidean(p1, p2):
    """
    Computes the Euclidean distance between two vectors.
    """
    return jnp.linalg.norm(p1 - p2)


@jax.jit
def is_in_bag(x: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
    """
    Checks if a point is already in the bag.
    """
    return (X == x).all(axis=1).any()


@partial(jax.jit, static_argnames="dist_fn")
def one_to_many_dists(x: jnp.ndarray, X: jnp.ndarray, dist_fn: Callable) -> jnp.ndarray:
    """
    Computes the distance between one point and many points.
    """
    dists_from_x_to_X = jax.vmap(dist_fn, in_axes=(None, 0))(x, X)
    return dists_from_x_to_X


@partial(jax.jit, static_argnames="dist_fn")
def pairwise_distances(X, dist_fn):
    """Compute pairwise distances between points in X using a custom distance function."""

    def x_to_X_dists(x):
        return jax.vmap(lambda y: dist_fn(x, y))(X)

    dists = jax.vmap(x_to_X_dists)(X)

    return dists


def submatrix(X: jnp.ndarray, remove_row: int, remove_col: int) -> jnp.ndarray:
    """
    Returns a submatrix of `X` removing the specified row and column.
    """
    return jnp.delete(
        jnp.delete(X, remove_row, axis=0, assume_unique_indices=True),
        remove_col,
        axis=1,
        assume_unique_indices=True,
    )


def matrix_cross_sum(X: jnp.ndarray, i: int, j: int, row_only=False):
    """
    Computes the sum of the elements in the row `i` and the column `j` of the matrix `X`.
    """
    X_row_i = X[i, :]
    if row_only:
        return X_row_i.sum()
    X_col_j = X[:, j]
    X_element_ij = X[i, j]
    return X_row_i.sum() + X_col_j.sum() - X_element_ij


def needless_point_idx(
    X: jnp.ndarray, dist_fn: Callable, potential_fn: Callable
) -> int:
    """
    Find a point in `X` removing which would decrease the total potential the most.
    """

    dists = pairwise_distances(X, dist_fn)
    potentials = jax.vmap(potential_fn)(dists)
    potentials = fill_diagonal(potentials, 0)  # replace diagonal elements with 0

    total_potentials_without_each_point = jax.vmap(
        lambda i: matrix_cross_sum(potentials, i, i, row_only=True)
    )(jnp.arange(X.shape[0]))

    # Compute the decrease in the total potential
    deltas = total_potentials_without_each_point

    # Find the point that would decrease the total potential the most (deltas are negative)
    idx = deltas.argmax()

    return idx


def _min_dist(x, X, dist_fn, n_valid, threshold=0.0):
    # # Initialize the distances with the first distance calculation
    dists = jnp.full((X.shape[0],), jnp.inf)

    # TODO: test both implementations: GPU, CPU, datasets

    def early_stop(dists):
        # Loop condition function
        def cond_fun(args):
            i, _dists, min_dist = args
            return (min_dist > threshold) & (i < n_valid)

        # Loop body function
        def body_fun(args):
            i, dists, min_dist = args
            disti = dist_fn(x, X[i])
            dists = dists.at[i].set(disti)
            min_dist = lax.min(min_dist, disti)
            return i + 1, dists, min_dist

        # Run the while loop
        _, dists, min_dist = lax.while_loop(cond_fun, body_fun, (0, dists, jnp.inf))

        return dists, min_dist

    def no_early_stop(dists):
        def body_fun(i, args):
            min_dist, dists = args
            disti = dist_fn(x, X[i])
            min_dist = lax.min(min_dist, disti)
            dists = dists.at[i].set(disti)
            return min_dist, dists

        min_dist, dists = lax.fori_loop(0, n_valid, body_fun, (jnp.inf, dists))
        return dists, min_dist

    dists, min_dist = lax.cond(
        threshold > 0.0,
        lambda: early_stop(dists),
        lambda: no_early_stop(dists),
    )

    is_above_threshold = min_dist > threshold
    return is_above_threshold, dists, min_dist


@partial(
    jax.jit,
    static_argnames=[
        "dist_fn",
        "k_neighbors",
        "threshold",
        "power",
        "approx_min",
        "n_valid_points",
    ],
)
def add_point_to_bag(
    x: jnp.ndarray,
    X: jnp.ndarray,
    dist_fn: Callable,
    k_neighbors: int,
    power: float,
    n_valid_points: int,
    threshold: float = 0.0,
    approx_min: bool = True,
) -> tuple[jnp.ndarray, bool, int]:
    """
    Adds one point to the bag, return the acceptance flag, the updated bag and the index
    of the replaced point (or -1 if no point was replaced).
    """

    assert k_neighbors > 0

    is_full = X.shape[0] == n_valid_points

    def below_threshold(X):
        return X, False, -1

    def above_threshold(X):
        def above_and_not_full(X):
            changed_item_idx = n_valid_points
            X = X.at[changed_item_idx].set(x)
            return X, True, changed_item_idx

        def above_and_full(X):
            # Find closest points in `X` to `x`
            # TODO: test approx_min_k vs argpartition
            if approx_min:
                k_closest_points_dists, k_closest_points_indices = lax.approx_min_k(
                    dists_from_x_to_X, k=k_neighbors
                )
            else:
                k_closest_points_dists, k_closest_points_indices = lax.top_k(
                    -dists_from_x_to_X, k_neighbors
                )
                k_closest_points_dists = -k_closest_points_dists

            # Define a neighborhood of `x`
            N = jnp.concatenate((jnp.array([x]), X[k_closest_points_indices]))

            # Find a point in `N` removing which would decrease the total potential the most
            needless_point_local_idx = needless_point_idx(
                N, dist_fn, lambda d: jnp.power(d, -power)
            )

            # If the needless point is not `x`, replace it with `x`
            is_accepted = needless_point_local_idx > 0
            changed_item_idx = k_closest_points_indices[needless_point_local_idx - 1]

            X, changed_item_idx = lax.cond(
                is_accepted,
                lambda X, idx: (X.at[changed_item_idx].set(x), idx),
                lambda X, idx: (X, -1),
                X,
                changed_item_idx,
            )

            return X, is_accepted, changed_item_idx

        result = lax.cond(
            is_full,
            lambda: above_and_full(X),
            lambda: above_and_not_full(X),
        )

        return result

    is_above_threshold, dists_from_x_to_X, min_dist = _min_dist(
        x, X, dist_fn, n_valid_points, threshold=threshold
    )

    X, is_accepted, updated_idx = lax.cond(
        is_above_threshold,
        above_threshold,
        below_threshold,
        X,
    )
    return X, is_accepted, updated_idx


@jax.jit
def finalize_updates(changes: jnp.ndarray) -> jnp.ndarray:
    """
    Given an array where each element represents whether a change occurred or
    not, -1 means no change, and a positive integer represents ID of changed
    object in some DB, return an array where only the last occurrence of each
    change is kept.

    Example:
    [ -1,  0,  2,  1,  1,  2,  2,  -1] <- identify unique changes
    [      0           1       2     ] <- only last changes are kept, -1s are ignored
    [ -1,  0, -1, -1,  1, -1,  2,  -1] <- all other elements are set to -1
    """
    # Take the last occurrence of each change
    changes_reversed = changes[::-1]
    _unique_changes, unique_idxs = jnp.unique(
        changes_reversed, return_index=True, size=changes.shape[0]
    )

    def keep_only_unique_change(i, changes_reversed):
        return lax.cond(
            jnp.isin(i, unique_idxs),
            lambda _: changes_reversed,
            lambda _: changes_reversed.at[i].set(-1),
            changes_reversed,
        )

    changes_reversed = lax.fori_loop(
        0, changes_reversed.shape[0], keep_only_unique_change, changes_reversed
    )

    return changes_reversed[::-1]


@partial(
    jax.jit,
    static_argnames=["dist_fn", "k_neighbors", "power", "threshold", "n_valid_points"],
)
def add_points_to_bag(
    *,
    X: jnp.ndarray,
    xs: jnp.ndarray,
    dist_fn: Callable,
    k_neighbors: int,
    threshold: float,
    power: float,
    n_valid_points: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    assert X.shape[0] >= n_valid_points
    assert xs.shape[0] > 0
    assert k_neighbors > 0
    assert power > 0
    assert threshold >= 0
    assert n_valid_points >= 0

    if X.dtype != xs.dtype:
        logger.warning(
            f"X and xs have different dtypes: {X.dtype} and {xs.dtype}, casting to {X.dtype}"
        )
        xs = xs.astype(X.dtype)

    # Initialize array to store the information about the changes
    changed_item_idxs = -jnp.ones(xs.shape[0], dtype=int)  # -1 means not changed

    def body_fun(i, args):
        X, changed_items_idxs, n_valid_points = args
        X_updated, is_accepted, changed_item_idx = add_point_to_bag(
            xs[i],
            X,
            dist_fn,
            k_neighbors=k_neighbors,
            threshold=threshold,
            power=power,
            n_valid_points=n_valid_points,
        )

        # Record the index of the replaced point
        changed_items_idxs = changed_items_idxs.at[i].set(changed_item_idx)

        n_valid_points = lax.cond(
            changed_item_idx == n_valid_points,
            lambda n: n + 1,
            lambda n: n,
            n_valid_points,
        )

        return X_updated, changed_items_idxs, n_valid_points

    X_new, changed_item_idxs, _ = lax.fori_loop(
        0, xs.shape[0], body_fun, (X, changed_item_idxs, n_valid_points)
    )

    # Some points might have been accepted and then replaced by another point
    changed_item_idxs = finalize_updates(changed_item_idxs)
    acceptance_mask = changed_item_idxs >= 0

    return changed_item_idxs, X_new, acceptance_mask
