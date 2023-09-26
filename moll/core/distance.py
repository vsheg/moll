import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def tanimoto(a: jnp.ndarray, b: jnp.ndarray) -> float:
    """
    Computes the Tanimoto distance between two vectors.
    """

    bitwise_and = jnp.sum(jnp.bitwise_and(a, b))
    bitwise_or = jnp.sum(jnp.bitwise_or(a, b))

    # Check for the case where both vectors are all zeros and return 0.0 in that case
    dist = jax.lax.cond(
        bitwise_or == 0,
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


def _is_in_bag(x: jnp.ndarray, X: jnp.ndarray) -> bool:
    """
    Checks if a point is already in the bag.
    """
    return (X == x).all(axis=1).any()


is_in_bag = jax.jit(_is_in_bag)


def _one_to_many_dists(x: jnp.ndarray, X: jnp.ndarray, dist: callable) -> jnp.ndarray:
    """
    Computes the distance between one point and many points.
    """
    dists_from_x_to_X = jax.vmap(dist, in_axes=(None, 0))(x, X)
    return dists_from_x_to_X


one_to_many_dists = jax.jit(_one_to_many_dists, static_argnames="dist")


def _add_point_to_bag(
    x: jnp.ndarray,
    X: jnp.ndarray,
    dist: callable,
    n_neighbors: int,
    threshold: float = 0.0,
    approx_min_k: bool = True,
) -> (jnp.ndarray, bool, int):
    """
    Adds one point to the bag, return the acceptance flag, the updated bag and the index of the replaced point (or -1
    if no point was replaced).
    """
    dists_from_x_to_X = _one_to_many_dists(x, X, dist)

    def below_threshold(X):
        return X, False, -1

    def above_threshold(X):
        # If there are less points in `X` than `n_neighbors`, use all points in `X`
        k_neigbour = min(n_neighbors, len(X))

        # Find closest points in `X` to `x`
        # TODO: test approx_min_k vs argpartition
        if approx_min_k:
            k_closest_points_dists, k_closest_points_indices = lax.approx_min_k(
                dists_from_x_to_X, k=k_neigbour
            )
        else:
            k_closest_points_indices = jnp.argpartition(dists_from_x_to_X, k_neigbour)[
                :k_neigbour
            ]
            k_closest_points_dists = dists_from_x_to_X[k_closest_points_indices]

        # Closest point:
        c_index = k_closest_points_indices[0]
        c = X[c_index]

        # Rest of the neighbours:
        rest_neighbours_indices = k_closest_points_indices[1:]
        N = X[rest_neighbours_indices]

        # Check that if we replace `c` with `x`, the distance to the rest of the points will increase
        repultion_x_to_rest_neighbours = jnp.power(k_closest_points_dists[1:], -2).sum()
        repultion_from_c_to_rest_neighbours = jnp.power(
            jax.vmap(dist, in_axes=(None, 0))(c, N), -2
        ).sum()

        is_accepted = (
            repultion_x_to_rest_neighbours < repultion_from_c_to_rest_neighbours
        )

        X = lax.cond(
            is_accepted,
            lambda X: X.at[c_index].set(x),
            lambda X: X,
            X,
        )

        return X, is_accepted, c_index

    X, is_accepted, c_index = lax.cond(
        dists_from_x_to_X.min() > threshold,
        above_threshold,
        below_threshold,
        X,
    )

    return X, is_accepted, c_index


add_point_to_bag = jax.jit(
    _add_point_to_bag,
    static_argnames=["dist", "n_neighbors", "threshold"],
)


def _keep_last_changes(changes: jnp.ndarray) -> jnp.ndarray:
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
    unique_changes, unique_idxs = jnp.unique(
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


keep_last_changes = jax.jit(_keep_last_changes)


def _add_points_to_bag(
    xs: jnp.ndarray,
    X: jnp.ndarray,
    dist: callable,
    n_neighbors: int,
    threshold: float,
) -> jnp.ndarray:
    # Initialize array to store the information about the changes
    changed_item_idxs = -jnp.ones(xs.shape[0], dtype=int)  # -1 means not altered

    def body_fun(i, args):
        X, altered_items_idxs = args
        X_new, _, altered_item_idx = _add_point_to_bag(
            xs[i], X, dist, n_neighbors=n_neighbors, threshold=threshold
        )

        # Record the index of the replaced point
        altered_items_idxs = altered_items_idxs.at[i].set(altered_item_idx)

        return X_new, altered_items_idxs

    X, changed_item_idxs = lax.fori_loop(
        0, xs.shape[0], body_fun, (X, changed_item_idxs)
    )
    # Some points might have been accepted and then replaced by another point
    changed_item_idxs = _keep_last_changes(changed_item_idxs)
    accepted_points_mask = changed_item_idxs >= 0

    return X, accepted_points_mask, changed_item_idxs


add_points_to_bag = _add_points_to_bag

add_points_to_bag = jax.jit(
    _add_points_to_bag,
    static_argnames=["dist", "n_neighbors", "threshold"],
)
