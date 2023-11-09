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


def pairwise_distances(X, dist_fn):
    """Compute pairwise distances between points in X using a custom distance function."""
    N = X.shape[0]

    # Create a function to compute distances from one point to all other points
    def single_point_distances(x):
        return jax.vmap(lambda y: dist_fn(x, y))(X)

    # Use vmap to compute pairwise distances for all points
    distances = jax.vmap(single_point_distances)(X)

    return distances


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


def find_nasty_point(X: jnp.ndarray, dist_fn: callable, potential_fn: callable) -> int:
    """
    Find a point in `X` removing which would decrease the total potential the most.
    """

    dists = pairwise_distances(X, dist_fn)
    potentials = jax.vmap(potential_fn)(dists) / 2  # divide by 2 because of symmetry
    potentials = jnp.nan_to_num(
        potentials, posinf=0
    )  # replace diagonal elements with 0
    total_potential = potentials.sum()

    # Drop each point and compute the total potential without it
    def body_fun(i):
        return submatrix(potentials, remove_row=i, remove_col=i).sum()

    total_potentials_without_each_point = jax.vmap(body_fun)(jnp.arange(X.shape[0]))

    # Compute the decrease in the total potential
    deltas = total_potentials_without_each_point - total_potential

    # Find the point that would decrease the total potential the most (deltas are negative)
    nasty_point_idx = deltas.argmin()

    return nasty_point_idx


def _add_point_to_bag(
    x: jnp.ndarray,
    X: jnp.ndarray,
    dist: callable,
    n_neighbors: int,
    power: float,
    threshold: float = 0.0,
    approx_min: bool = True,
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
        if approx_min:
            k_closest_points_dists, k_closest_points_indices = lax.approx_min_k(
                dists_from_x_to_X, k=k_neigbour
            )
        else:
            k_closest_points_dists, k_closest_points_indices = lax.top_k(
                -dists_from_x_to_X, k_neigbour
            )
            k_closest_points_dists = -k_closest_points_dists

        # Define a neighborhood of `x`
        N = jnp.concatenate((jnp.array([x]), X[k_closest_points_indices]))

        # Find a point in `N` removing which would decrease the total potential the most
        nasty_point_local_idx = find_nasty_point(
            N, dist, lambda d: jnp.power(d, -power)
        )

        # If the nasty point is not `x`, replace it with `x`
        is_accepted = nasty_point_local_idx > 0
        replace_idx = k_closest_points_indices[nasty_point_local_idx - 1]

        X = lax.cond(
            is_accepted,
            lambda X: X.at[replace_idx].set(x),
            lambda X: X,
            X,
        )

        return X, is_accepted, k_closest_points_indices[nasty_point_local_idx - 1]

    X, is_accepted, replace_idx = lax.cond(
        dists_from_x_to_X.min() > threshold,
        above_threshold,
        below_threshold,
        X,
    )

    return X, is_accepted, replace_idx


add_point_to_bag = jax.jit(
    _add_point_to_bag,
    static_argnames=["dist", "n_neighbors", "threshold", "power", "approx_min"],
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
    power: float,
) -> jnp.ndarray:
    # Initialize array to store the information about the changes
    changed_item_idxs = -jnp.ones(xs.shape[0], dtype=int)  # -1 means not altered

    def body_fun(i, args):
        X, altered_items_idxs = args
        X_new, _, altered_item_idx = _add_point_to_bag(
            xs[i], X, dist, n_neighbors=n_neighbors, threshold=threshold, power=power
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


add_points_to_bag = jax.jit(
    _add_points_to_bag,
    static_argnames=["dist", "n_neighbors", "power", "threshold"],
)
