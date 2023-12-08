import time
from collections.abc import Sequence
from functools import partial
from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax import Array, lax

__all__ = [
    "time_int_seed",
    "time_key",
    "create_key",
    "points_around",
    "grid_centers",
    "globs",
    "random_grid_points",
    "partition",
    "fill_diagonal",
    "dist_matrix",
    "dists_to_nearest_neighbor",
]

Seed: TypeAlias = int | jnp.ndarray | None


def time_int_seed() -> int:
    """Returns a number of microseconds since the epoch."""
    return int(time.time() * 1e6)


def time_key() -> jnp.ndarray:
    """Returns a JAX PRNG key based on the current time."""
    return jax.random.PRNGKey(time_int_seed())


def create_key(seed: Seed = None) -> jnp.ndarray:
    """
    Create a JAX PRNG key from a seed.
    """
    if seed is None:
        return time_key()
    if isinstance(seed, int):
        return jax.random.PRNGKey(seed)
    return seed


def cap_vector(vector: Array, max_length: float):
    """Truncate a vector to a maximum length."""
    length = jnp.linalg.norm(vector)

    return lax.cond(
        length <= max_length,
        lambda v: v,
        lambda v: v / length * max_length,
        vector,
    )


def points_around(
    center: jnp.ndarray,
    n_points: int,
    std: float = 1,
    cap_radius: float | None = None,
    seed: Seed = None,
):
    """Generate points around a center."""
    key = create_key(seed)
    dim = len(center)
    offsets = jax.random.normal(key, shape=(n_points, dim)) * std
    if cap_radius is not None:
        offsets = jax.vmap(cap_vector, in_axes=(0, None))(offsets, cap_radius)
    return center + offsets


def grid_centers(
    n_ticks: tuple[int, ...],
    range: float = 1,
    std: float = 0.0,
    seed: Seed = None,
) -> jnp.ndarray:
    """
    Generate grid centers.
    """
    dim = len(n_ticks)
    ticks = jnp.linspace(-1, 1, num=n_ticks[0]) * range
    centers = jnp.array(jnp.meshgrid(*[ticks] * dim)).reshape(dim, -1).T
    if std > 0:
        key = create_key(seed)
        offsets = jax.random.normal(key, shape=centers.shape) * std
        centers += offsets
    return centers


def globs(
    centers: Array,
    sizes: Sequence[int] | int = 10,
    stds: Sequence[float] | float = 1,
    seed: Seed = None,
    cap_radius: float | None = None,
    shuffle=True,
):
    """Generate points around centers."""
    if isinstance(sizes, int):
        sizes = (sizes,) * len(centers)

    if isinstance(stds, float | int):
        stds = (stds,) * len(centers)

    key = create_key(seed)

    n_centers = len(centers)
    keys = jax.random.split(key, n_centers)
    points = []

    for center, size, std, key in zip(centers, sizes, stds, keys, strict=True):
        ps = points_around(
            center, n_points=size, std=std, cap_radius=cap_radius, seed=key
        )
        points.append(ps)

    points = jnp.concatenate(points, axis=0)

    if shuffle:
        subkey, subkey = jax.random.split(key)
        points = jax.random.permutation(subkey, points)
    return points


def random_grid_points(
    n_points: int,
    dim: int,
    n_ticks: int,
    spacing: int = 1,
    seed: Seed = None,
):
    """
    Generate random grid points.
    """
    ticks = spacing * jnp.arange(n_ticks) - n_ticks / 2
    grid = jnp.stack(jnp.meshgrid(*[ticks] * dim), axis=-1).reshape(-1, dim)
    # TODO: do not generate all possible grid points, just sample

    key = create_key(seed)
    sample = jax.random.choice(key, grid, shape=(n_points,), replace=False)
    return sample


def partition(data: list, *, n_partitions: int):
    """
    Partition the data into `n_partitions` partitions.
    """

    n_points = len(data)
    partition_size = n_points // n_partitions + 1
    partitions = [
        data[i : i + partition_size] for i in range(0, n_points, partition_size)
    ]
    return partitions


def fill_diagonal(array: jnp.ndarray, val: float | int):
    assert array.ndim >= 2
    i, j = jnp.diag_indices(min(array.shape[-2:]))
    return array.at[..., i, j].set(val)


@partial(jax.jit, static_argnames="dist_fn")
def dist_matrix(points, dist_fn):
    """Compute pairwise distances between points."""
    expanded_points = jnp.expand_dims(points, axis=1)
    distances = jax.vmap(jax.vmap(dist_fn, in_axes=(None, 0)), in_axes=(0, None))(
        points, expanded_points
    )
    return distances


@partial(jax.jit, static_argnames="dist_fn")
def dists_to_nearest_neighbor(points, dist_fn):
    """Compute pairwise distances between points."""
    dists_ = dist_matrix(points, dist_fn)
    dists_ = fill_diagonal(dists_, jnp.inf)
    return jnp.min(dists_, axis=0)
