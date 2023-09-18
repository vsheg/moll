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


def _add_one_point_to_the_bag(
    x: jnp.ndarray,
    X: jnp.ndarray,
    dist: callable,
    n_neighbors: int,
    threshold: float = 0.0,
) -> (bool, jnp.ndarray):
    dists_from_x_to_X = _one_to_many_dists(x, X, dist)

    def below_threshold(X):
        return False, X

    def above_threshold(X):
        # If there are less points in `X` than `n_neighbors`, use all points in `X`
        k_neigbour = min(n_neighbors, len(X))

        # Find closest points in `X` to `x`
        k_closest_points_dists, k_closest_points_indices = lax.approx_min_k(
            dists_from_x_to_X, k=k_neigbour
        )

        # Closest point:
        c_index = k_closest_points_indices[0]
        c = X[c_index]

        # Rest of the neighbours:
        rest_neighbours_indices = k_closest_points_indices[1:]
        N = X[rest_neighbours_indices]

        # Check that if we replace `c` with `x`, the distance to the rest of the points will increase
        repultion_x_to_rest_neighbours = jnp.power(k_closest_points_dists[1:], -3).sum()
        repultion_from_c_to_rest_neighbours = jnp.power(
            jax.vmap(dist, in_axes=(None, 0))(c, N), -3
        ).sum()

        is_accepted = (
            repultion_x_to_rest_neighbours < repultion_from_c_to_rest_neighbours
        )

        X = lax.cond(
            is_accepted,
            lambda _: X.at[c_index].set(x),
            lambda _: X,
            None,
        )

        return is_accepted, X

    is_accepted, X = lax.cond(
        dists_from_x_to_X.min() > threshold,
        above_threshold,
        below_threshold,
        X,
    )

    return is_accepted, X


add_one_point_to_the_bag = jax.jit(
    _add_one_point_to_the_bag, static_argnames=["dist", "n_neighbors", "threshold"]
)


def _add_points_to_bag(
    xs: jnp.ndarray,
    X: jnp.ndarray,
    dist: callable,
    n_neighbors: int,
    threshold: float,
) -> jnp.ndarray:
    acceptances = jnp.zeros(xs.shape[0], dtype=bool)

    def body_fun(i, args):
        acceptances, X = args
        is_accepted, X_new = _add_one_point_to_the_bag(
            xs[i], X, dist, n_neighbors=n_neighbors, threshold=threshold
        )
        acceptances = acceptances.at[i].set(is_accepted)
        return acceptances, X_new

    acceptances, X = lax.fori_loop(0, xs.shape[0], body_fun, (acceptances, X))

    return acceptances, X


add_points_to_bag = jax.jit(
    _add_points_to_bag, static_argnames=["dist", "n_neighbors", "threshold"]
)
