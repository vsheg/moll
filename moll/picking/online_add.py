"""
Online algorithm for adding points to a fixed-size set of points.
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax

from ..utils import dist_matrix, fill_diagonal, matrix_cross_sum


@partial(jax.jit, static_argnames=["similarity_fn", "potential_fn"])
def _needless_point_idx(
    vicinity: Array, similarity_fn: Callable, potential_fn: Callable
) -> int:
    """
    Find a point in `X` removing which would decrease the total potential the most.
    """
    # Calculate matrix of pairwise distances and corresponding matrix of potentials
    dist_mat = dist_matrix(vicinity, similarity_fn)
    potent_mat = jax.vmap(potential_fn)(dist_mat)

    # Inherent potential of a point is 0, it means that the point by itself does not
    # contribute to the total potential. In the future, the penalty for the point itself
    # can be added
    potent_mat = fill_diagonal(potent_mat, 0)

    # Compute potential decrease when each point is deleted from the set
    deltas = jax.vmap(lambda i: matrix_cross_sum(potent_mat, i, i, row_only=True))(
        jnp.arange(vicinity.shape[0])
    )

    # Find point that decreases total potential the most (deltas are negative)
    idx = deltas.argmax()
    return idx


def _similarities(x, X, similarity_fn, n_valid):
    def sim_i(i):
        return lax.cond(
            i < n_valid,
            lambda i: similarity_fn(x, X[i]),
            lambda _: jnp.inf,
            i,
        )

    return jax.vmap(sim_i)(jnp.arange(X.shape[0]))


def _k_neighbors(similarities: Array, k_neighbors: int):
    k_neighbors_idxs = lax.approx_min_k(similarities, k=k_neighbors)[1]
    return k_neighbors_idxs


@partial(
    jax.jit,
    static_argnames=["similarity_fn", "potential_fn", "k_neighbors"],
    donate_argnames=["x", "X"],
    inline=True,
)
def _add_point(
    x: Array,
    X: Array,
    similarity_fn: Callable,
    potential_fn: Callable,
    k_neighbors: int,
    n_valid_points: int,
    threshold: float,
) -> tuple[Array, bool, int]:
    """
    Adds a point `x` to a fixed-size set of points `X`.
    """

    def below_threshold_or_infinite_potential(X, _):
        return X, False, -1

    def above_threshold_and_not_full(X, _):
        updated_point_idx = n_valid_points
        X = X.at[updated_point_idx].set(x)
        return X, True, updated_point_idx

    def above_threshold_and_full(X, dists):
        # TODO: test approx_min_k vs argpartition

        k_neighbors_idxs = _k_neighbors(dists, k_neighbors=k_neighbors)

        # Define a neighborhood of `x`
        vicinity = lax.concatenate((jnp.array([x]), X[k_neighbors_idxs]), 0)

        # Point in the vicinity removing which decreases the total potential the most:
        needless_point_vicinity_idx = _needless_point_idx(
            vicinity, similarity_fn, potential_fn
        )

        # If the needless point is not `x`, replace it with `x`
        is_accepted = needless_point_vicinity_idx > 0
        updated_point_idx = k_neighbors_idxs[needless_point_vicinity_idx - 1]

        X, updated_point_idx = lax.cond(
            is_accepted,
            lambda X, idx: (X.at[updated_point_idx].set(x), idx),
            lambda X, _: (X, -1),
            X,
            updated_point_idx,
        )

        return X, is_accepted, updated_point_idx

    is_full = X.shape[0] == n_valid_points

    dists = _similarities(x, X, similarity_fn, n_valid_points)
    min_dist = dists.min()
    is_above_threshold = min_dist > threshold

    branches = [
        below_threshold_or_infinite_potential,
        above_threshold_and_not_full,
        above_threshold_and_full,
    ]

    branch_idx = 0 + (is_above_threshold) + (is_full & is_above_threshold)

    # If the potential is infinite, the point is always rejected
    is_potential_infinite = jnp.isinf(potential_fn(min_dist))
    branch_idx *= ~is_potential_infinite

    result = X, is_accepted, updated_point_idx = lax.switch(
        branch_idx, branches, X, dists
    )
    return result


@partial(jax.jit, donate_argnames=["changes"], inline=True)
def _finalize_updates(changes: Array) -> Array:
    """
    Given an array where each element represents whether a change occurred or
    not, -1 means no change, and a positive integer represents ID of updated
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

    idxs = jnp.arange(changes.shape[0])
    mask = jnp.isin(idxs, unique_idxs)[::-1]
    return jnp.where(mask, changes, -1)


@partial(
    jax.jit,
    static_argnames=["similarity_fn", "potential_fn", "k_neighbors"],
    donate_argnames=["xs", "X"],
)
def update_points(
    *,
    X: Array,
    xs: Array,
    similarity_fn: Callable,
    potential_fn: Callable,
    k_neighbors: int,
    threshold: float,
    n_valid_points: int,
) -> tuple[Array, Array, Array]:
    assert xs.shape[0] > 0
    assert X.dtype == xs.dtype

    # Initialize array to store the information about the changes
    updated_points_idxs = -jnp.ones(xs.shape[0], dtype=int)  # -1 means not updated

    def body_fun(i, args):
        X, updated_points_idxs, n_valid_points = args
        X_updated, is_accepted, updated_point_idx = _add_point(
            xs[i],
            X,
            similarity_fn=similarity_fn,
            potential_fn=potential_fn,
            k_neighbors=k_neighbors,
            threshold=threshold,
            n_valid_points=n_valid_points,
        )

        # Record the index of the replaced point
        updated_points_idxs = updated_points_idxs.at[i].set(updated_point_idx)

        n_valid_points = lax.cond(
            updated_point_idx == n_valid_points,
            lambda n: n + 1,
            lambda n: n,
            n_valid_points,
        )

        return X_updated, updated_points_idxs, n_valid_points

    X_new, updated_points_idxs, _ = lax.fori_loop(
        0, xs.shape[0], body_fun, (X, updated_points_idxs, n_valid_points)
    )

    # Some points might have been accepted and then replaced by another point
    updated_points_idxs = _finalize_updates(updated_points_idxs)
    acceptance_mask = updated_points_idxs >= 0

    return updated_points_idxs, X_new, acceptance_mask
