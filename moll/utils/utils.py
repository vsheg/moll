import time
from typing import TypeAlias

import jax
import jax.numpy as jnp

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


def points_around(
    center: jnp.ndarray,
    n_points: int,
    seed: Seed = None,
    std: float = 1,
):
    """Generate points around a center."""
    key = create_key(seed)
    dim = len(center)
    offsets = jax.random.normal(key, shape=(n_points, dim)) * std
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
    centers: jnp.array,
    sizes: tuple[int, ...] | int = 10,
    stds: tuple[int, ...] | int = 1,
    seed: Seed = None,
    shuffle=True,
):
    """Generate points around centers."""
    if isinstance(sizes, int):
        sizes = (sizes,) * len(centers)

    if isinstance(stds, int):
        stds = (stds,) * len(centers)

    key = create_key(seed)

    n_centers = len(centers)
    keys = jax.random.split(key, n_centers)
    points = []

    for center, size, std, key in zip(centers, sizes, stds, keys, strict=True):
        ps = points_around(center, n_points=size, std=std, seed=key)
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
