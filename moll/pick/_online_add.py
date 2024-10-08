"""
Online algorithm for adding vectors to a fixed-size set of vectors.
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax

from ..utils import dist_matrix, fill_diagonal, matrix_cross_sum


@partial(jax.jit, static_argnames=["dist_fn", "sim_fn", "loss_fn", "maximize"])
def _needless_vector_idx(
    vicinity: Array,
    dist_fn: Callable[[Array, Array], Array],
    sim_fn: Callable[[Array], Array],
    loss_fn: Callable[[Array], Array],
    maximize: bool = False,
) -> int:
    """
    Find a vector in `X` removing which would decrease the total potential the most.
    """
    # Calculate matrix of pairwise distances and corresponding matrix of potentials
    dist_mat = dist_matrix(vicinity, dist_fn)
    sim_mat = jax.vmap(sim_fn)(dist_mat)
    loss_mat = jax.vmap(loss_fn)(sim_mat)

    # Inherent potential of a vector is 0, it means that the vector by itself does not
    # contribute to the total potential. In the future, the penalty for the vector itself
    # can be added
    loss_mat = fill_diagonal(loss_mat, 0)

    # Compute potential decrease when each vector is deleted from the set
    deltas = jax.vmap(lambda i: matrix_cross_sum(loss_mat, i, i, row_only=True))(
        jnp.arange(vicinity.shape[0])
    )

    # If the goal is to maximize the potential, return the vector with the largest delta
    if maximize:
        deltas = -deltas

    # Find vector that decreases total potential the most (deltas are negative)
    idx = deltas.argmax()

    return idx


@partial(jax.jit, static_argnames=["dist_fn", "sim_fn"])
def _similarities(x: Array, X: Array, dist_fn: Callable, sim_fn: Callable, n_valid: int):
    def sim_i(i):
        return lax.cond(
            i < n_valid,
            lambda i: sim_fn(dist_fn(x, X[i])),
            lambda _: jnp.nan,
            i,
        )

    return jax.vmap(sim_i)(jnp.arange(X.shape[0]))


@partial(jax.jit, static_argnames=["k_neighbors"])
def _k_neighbors(similarities: Array, k_neighbors: int):
    # TODO: test approx_min_k vs argpartition
    k_neighbors_idxs = lax.approx_min_k(similarities, k=k_neighbors)[1]
    return k_neighbors_idxs


@partial(
    jax.jit,
    static_argnames=[
        "dist_fn",
        "sim_fn",
        "loss_fn",
        "k_neighbors",
        "sim_min",
        "sim_max",
        "maximize",
    ],
    donate_argnames=["x", "X"],
    inline=True,
)
def _add_vector(
    x: Array,
    X: Array,
    dist_fn: Callable[[Array, Array], Array],
    sim_fn: Callable[[Array], Array],
    loss_fn: Callable[[Array], Array],
    k_neighbors: int,
    n_valid_vectors: int,
    sim_min: float = -jnp.inf,
    sim_max: float = +jnp.inf,
    X_pinned: Array | None = None,
    maximize: bool = False,
    reject_inf: bool = True,
) -> tuple[Array, int]:
    """
    Adds a vector `x` to a fixed-size set of vectors `X`.
    """

    # Prepend pinned vectors to the data
    if n_pinned := (X_pinned.shape[0] if X_pinned is not None else 0):
        # TODO: test the case when dtypes of X and X_pinned are different.
        X = jnp.concatenate((X_pinned, X), 0)  # type: ignore

    n_valid_vectors += n_pinned
    min_changeable_idx = n_pinned

    def outside_sim_limits_or_loss_positive_inf(X, _):
        return X, -1

    def within_sim_limits_and_not_full(X, _):
        updated_vector_idx = n_valid_vectors
        X = X.at[updated_vector_idx].set(x)
        return X, updated_vector_idx

    def within_sim_limits_and_full(X, sims):
        # Find `k_neighbors` most similar vectors to `x`
        k_neighbors_idxs = _k_neighbors(sims, k_neighbors=k_neighbors)

        # Define a neighborhood of `x`
        vicinity = lax.concatenate((jnp.array([x]), X[k_neighbors_idxs]), 0)

        # Vector in the vicinity removing which decreases the total potential the most:
        needless_vector_vicinity_idx = _needless_vector_idx(
            vicinity, dist_fn, sim_fn, loss_fn, maximize=maximize
        )

        # If the needless vector is not `x`, replace it with `x`
        is_accepted = needless_vector_vicinity_idx > 0
        needless_vector_idx = k_neighbors_idxs[needless_vector_vicinity_idx - 1]

        is_accepted *= needless_vector_idx >= min_changeable_idx

        X, needless_vector_idx = lax.cond(
            is_accepted,
            lambda X, idx: (X.at[needless_vector_idx].set(x), idx),
            lambda X, _: (X, -1),
            *(X, needless_vector_idx),
        )

        return X, needless_vector_idx

    is_full = X.shape[0] == n_valid_vectors

    sims = _similarities(x, X, dist_fn, sim_fn, n_valid_vectors)
    losses = jax.vmap(loss_fn)(sims)

    has_loss_positive_inf = jnp.nanmax(losses) == jnp.inf
    is_within_sim_limits = (sim_min <= jnp.nanmin(sims)) & (jnp.nanmax(sims) <= sim_max)

    branches = [
        outside_sim_limits_or_loss_positive_inf,
        within_sim_limits_and_not_full,
        within_sim_limits_and_full,
    ]

    branch_idx = 0 + (is_within_sim_limits) + (is_within_sim_limits & is_full)
    branch_idx *= ~(has_loss_positive_inf * reject_inf)

    X, updated_vector_idx = lax.switch(branch_idx, branches, X, sims)

    # Make a correction for the prepended pinned vectors
    index_correction = jnp.where(updated_vector_idx >= n_pinned, n_pinned, 0)
    updated_vector_idx -= index_correction

    return X[n_pinned:], updated_vector_idx


@partial(jax.jit, donate_argnames=["changes"], inline=True)
def _finalize_updates(changes: Array) -> Array:
    """
    Remove all but the last occurrence of each change, set all other elements to -1.

    Given an array where each element represents whether a change occurred or
    not, -1 means "no change", and a positive integer represents ID of an updated
    object in some DB, return an array where only the last occurrence of each
    change is kept.

    Note:
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
    static_argnames=["dist_fn", "sim_fn", "loss_fn", "k_neighbors", "maximize"],
    donate_argnames=["X", "X_pinned", "xs"],
)
def update_vectors(
    *,
    X: Array,
    xs: Array,
    dist_fn: Callable,
    sim_fn: Callable,
    loss_fn: Callable,
    k_neighbors: int,
    sim_min: float,
    sim_max: float,
    n_valid: int,
    X_pinned: Array | None = None,
    maximize: bool = False,
) -> tuple[Array, Array, Array, int, int]:
    assert xs.shape[0] > 0
    # assert X.dtype == xs.dtype # TODO: fix dtype

    # Initialize array to store the information about the changes
    updated_idxs = -jnp.ones(xs.shape[0], dtype=int)  # -1 means not updated

    def body_fun(carry, x):
        X, n_valid_new = carry

        X, updated_idx = _add_vector(
            x,
            X,
            X_pinned=X_pinned,
            dist_fn=dist_fn,
            sim_fn=sim_fn,
            loss_fn=loss_fn,
            k_neighbors=k_neighbors,
            sim_min=sim_min,
            sim_max=sim_max,
            n_valid_vectors=n_valid_new,
            maximize=maximize,
        )

        n_valid_new = lax.cond(
            updated_idx == n_valid_new,
            lambda n: n + 1,
            lambda n: n,
            n_valid_new,
        )

        return (X, n_valid_new), updated_idx

    (X, n_valid_new), updated_idxs = lax.scan(body_fun, (X, n_valid), xs)

    # Some vectors might have been accepted and then replaced by another vector
    updated_idxs = _finalize_updates(updated_idxs)
    acceptance_mask = updated_idxs >= 0

    n_appended = n_valid_new - n_valid
    n_updated = ((updated_idxs >= 0) & (updated_idxs < n_valid)).sum()

    return X, updated_idxs, acceptance_mask, n_appended, n_updated
