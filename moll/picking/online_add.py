from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax

from ..metrics.utils import _matrix_cross_sum, _pairwise_distances
from ..utils.utils import fill_diagonal


def _needless_point_idx(
    X: jnp.ndarray, dist_fn: Callable, potential_fn: Callable
) -> int:
    """
    Find a point in `X` removing which would decrease the total potential the most.
    """

    dists = _pairwise_distances(X, dist_fn)
    potentials = jax.vmap(potential_fn)(dists)
    potentials = fill_diagonal(potentials, 0)  # replace diagonal elements with 0

    # Compute potential decrease for each point
    deltas = jax.vmap(lambda i: _matrix_cross_sum(potentials, i, i, row_only=True))(
        jnp.arange(X.shape[0])
    )

    # Find point that decreases total potential the most (deltas are negative)
    idx = deltas.argmax()

    return idx


def dists(x, X, dist_fn, n_valid, threshold=0.0):
    ds = jax.vmap(dist_fn, in_axes=(None, 0))(x, X)
    mask = jnp.arange(X.shape[0]) < n_valid
    ds = jnp.where(mask, ds, jnp.inf)
    return ds.min() > threshold, ds, ds.min()


def _add_point(
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

    def below_threshold(X):
        return X, False, -1

    def above_and_not_full(X):
        changed_item_idx = n_valid_points
        X = X.at[changed_item_idx].set(x)
        return X, True, changed_item_idx

    def above_and_full(X):
        # Find closest points in `X` to `x`
        # TODO: test approx_min_k vs argpartition

        k_closest_points_dists, k_closest_points_indices = lax.cond(
            approx_min,
            lambda ds: lax.approx_min_k(ds, k=k_neighbors),
            lambda ds: lax.top_k(-ds, k=k_neighbors),
            dists_from_x_to_X,
        )

        k_closest_points_dists = lax.abs(k_closest_points_dists)  # for top_k

        # Define a neighborhood of `x`
        N = jnp.concatenate((jnp.array([x]), X[k_closest_points_indices]))

        # Find a point in `N` removing which would decrease the total potential the most
        needless_point_local_idx = _needless_point_idx(N, dist_fn, lambda d: d**-power)

        # If the needless point is not `x`, replace it with `x`
        is_accepted = needless_point_local_idx > 0
        changed_item_idx = k_closest_points_indices[needless_point_local_idx - 1]

        X, changed_item_idx = lax.cond(
            is_accepted,
            lambda X, idx: (X.at[changed_item_idx].set(x), idx),
            lambda X, _: (X, -1),
            X,
            changed_item_idx,
        )

        return X, is_accepted, changed_item_idx

    is_full = X.shape[0] == n_valid_points

    is_above_threshold, dists_from_x_to_X, min_dist = dists(
        x, X, dist_fn, n_valid_points, threshold=threshold
    )

    branches = [below_threshold, above_and_not_full, above_and_full]
    branch_idx = 0 + is_above_threshold + (is_full & is_above_threshold)

    return lax.switch(branch_idx, branches, X)


def _finalize_updates(changes: jnp.ndarray) -> jnp.ndarray:
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

    idxs = jnp.arange(changes.shape[0])
    mask = jnp.isin(idxs, unique_idxs)[::-1]
    return jnp.where(mask, changes, -1)


def _add_points(
    *,
    X: jnp.ndarray,
    xs: jnp.ndarray,
    dist_fn: Callable,
    k_neighbors: int,
    threshold: float,
    power: float,
    n_valid_points: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    assert xs.shape[0] > 0
    assert X.dtype == xs.dtype

    # Initialize array to store the information about the changes
    changed_item_idxs = -jnp.ones(xs.shape[0], dtype=int)  # -1 means not changed

    def body_fun(i, args):
        X, changed_items_idxs, n_valid_points = args
        X_updated, is_accepted, changed_item_idx = _add_point(
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
    changed_item_idxs = _finalize_updates(changed_item_idxs)
    acceptance_mask = changed_item_idxs >= 0

    return changed_item_idxs, X_new, acceptance_mask


add_points = jax.jit(
    _add_points,
    static_argnames=["dist_fn", "k_neighbors"],
    donate_argnames=["X"],
)