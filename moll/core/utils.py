import itertools

import jax
import jax.numpy as jnp


def points_around(
    center: jnp.ndarray,
    n_points: int,
    key: jnp.ndarray,
    radius: float | int = 1,
):
    dim = len(center)
    offsets = jax.random.uniform(
        key, shape=(n_points, dim), minval=-radius, maxval=radius
    )
    return center + offsets


def generate_points(
    centers: jnp.ndarray,
    sizes: list,
    seed: int,
    radius: float | int = 1,
    shuffle: bool = True,
) -> jnp.ndarray:
    """
    Generate points around centers.
    """
    assert len(centers) == len(sizes), "Number of centers and sizes must be the same"

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, len(centers))

    points = jnp.concatenate(
        [
            points_around(center, n_points=size, radius=radius, key=key)
            for center, size, key in zip(centers, sizes, keys)
        ],
        axis=0,
    )

    if shuffle:
        key, subkey = jax.random.split(key)
        points = jax.random.permutation(subkey, points)

    return points


def random_grid_points(
    n_points: int,
    dim: int,
    n_ticks: int,
    seed: int,
    spacing: int = 1,
):
    """
    Generate random grid points.
    """
    ticks = spacing * jnp.arange(n_ticks) - n_ticks / 2
    grid = jnp.stack(jnp.meshgrid(*[ticks] * dim), axis=-1).reshape(-1, dim)
    # TODO: do not generate all possible grid points, just sample

    key = jax.random.PRNGKey(seed)
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
