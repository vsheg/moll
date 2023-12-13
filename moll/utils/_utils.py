import time
from collections.abc import Generator, Sequence
from functools import partial
from pathlib import Path
from typing import TypeAlias

import jax
import jax.numpy as jnp
from jax import Array, lax
from public import public

from ._decorators import listify

Seed: TypeAlias = int | Array | None


@public
def time_int_seed() -> int:
    """Returns a number of microseconds since the epoch."""
    return int(time.time() * 1e6)


@public
def time_key() -> Array:
    """Returns a JAX PRNG key based on the current time."""
    return jax.random.PRNGKey(time_int_seed())


@public
def create_key(seed: Seed = None) -> Array:
    """
    Create a JAX PRNG key from a seed.
    """
    if seed is None:
        return time_key()
    if isinstance(seed, int):
        return jax.random.PRNGKey(seed)
    return seed


@public
def cap_vector(vector: Array, max_length: float):
    """Truncate a vector to a maximum length."""
    length = jnp.linalg.norm(vector)

    return lax.cond(
        length <= max_length,
        lambda v: v,
        lambda v: v / length * max_length,
        vector,
    )


@public
def points_around(
    center: Array,
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
) -> Array:
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


@public
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


@public
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


@public
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


@public
def fill_diagonal(array: Array, val: float | int):
    assert array.ndim >= 2
    i, j = jnp.diag_indices(min(array.shape[-2:]))
    return array.at[..., i, j].set(val)


@public
@partial(jax.jit, static_argnames=["dist_fn", "condensed"])
def dist_matrix(points, dist_fn, condensed=False):
    """
    Compute pairwise distances between points.

    Examples:
        >>> points = jnp.array([[0, 0], [1, 0]])
        >>> dist_fn = lambda x, y: jnp.linalg.norm(x - y)

        >>> dist_matrix(points, dist_fn).tolist()
        [[0.0, 1.0], [1.0, 0.0]]

        >>> dist_matrix(points, dist_fn, condensed=True).tolist()
        [1.0]
    """
    dists = jax.vmap(jax.vmap(dist_fn, in_axes=(None, 0)), in_axes=(0, None))(
        points, points
    )

    if condensed:
        size = points.shape[0]
        indices = jnp.triu_indices(size, k=1)
        dists = dists[indices]

    return dists


@public
@partial(jax.jit, static_argnames=["i", "j", "row_only", "crossover"])
def matrix_cross_sum(X: Array, i: int, j: int, row_only=False, crossover=True):
    """
    Compute the sum of the elements in the row `i` and the column `j` of the matrix `X`.
    """
    X = lax.cond(
        crossover,
        lambda X: X,
        lambda X: X.at[i, j].set(0),
        X,
    )

    return lax.cond(
        row_only,
        lambda X: X[i, :].sum(),
        lambda X: X[i, :].sum() + X[:, j].sum() - X[i, j],
        X,
    )


@public
@partial(jax.jit, static_argnames="dist_fn")
def dists_to_nearest_neighbor(points, dist_fn):
    """Compute pairwise distances between points."""
    dists_ = dist_matrix(points, dist_fn)
    dists_ = fill_diagonal(dists_, jnp.inf)
    return jnp.min(dists_, axis=0)


@listify
def group_files_by_size(
    files: list[Path], max_batches: int, *, sort_size=True, large_first=False
) -> Generator[list[Path], None, None]:
    """
    Greedily groups files into batches by their size.

    Note:
        Number of batches may be less than `max_batches`.
    """
    if max_batches <= 0:
        raise ValueError("`max_batches` must be greater than 0")

    max_batches = min(max_batches, len(files))

    files_with_byte_sizes = [(f, f.stat().st_size) for f in files]
    total_byte_size = sum(size for _, size in files_with_byte_sizes)

    batch_byte_size_threshold = max(1, total_byte_size // max_batches + 1)

    if sort_size:
        files_with_byte_sizes = sorted(
            files_with_byte_sizes, key=lambda tup: tup[1], reverse=(not large_first)
        )

    batch = []
    batch_byte_size = 0

    while files_with_byte_sizes:
        file, byte_size = files_with_byte_sizes.pop()
        batch.append(file)
        batch_byte_size += byte_size

        if batch_byte_size >= batch_byte_size_threshold:
            yield batch
            batch = []
            batch_byte_size = 0

    if batch:
        yield batch
