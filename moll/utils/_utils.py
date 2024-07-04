import time
from collections.abc import Callable, Generator, Sequence
from functools import partial
from pathlib import Path
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.typing import ArrayLike, DTypeLike
from numpy.typing import DTypeLike as NPDTypeLike
from numpy.typing import NDArray
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
def vectors_around(
    center: Array,
    n_vectors: int,
    std: float = 1,
    cap_radius: float | None = None,
    seed: Seed = None,
):
    """Generate vectors around a central vector."""
    key = create_key(seed)
    dim = len(center)
    offsets = jax.random.normal(key, shape=(n_vectors, dim)) * std
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
    """Generate vectors around centers."""
    if isinstance(sizes, int):
        sizes = (sizes,) * len(centers)

    if isinstance(stds, float | int):
        stds = (stds,) * len(centers)

    key = create_key(seed)

    n_centers = len(centers)
    keys = jax.random.split(key, n_centers)
    vectors = []

    for center, size, std, key in zip(centers, sizes, stds, keys, strict=True):
        ps = vectors_around(
            center, n_vectors=size, std=std, cap_radius=cap_radius, seed=key
        )
        vectors.append(ps)

    vectors = jnp.concatenate(vectors, axis=0)

    if shuffle:
        subkey, subkey = jax.random.split(key)
        vectors = jax.random.permutation(subkey, vectors)
    return vectors


@public
def random_grid_points(
    n_points: int,
    dim: int,
    n_ticks: int,
    spacing: int = 1,
    seed: Seed = None,
):
    """
    Generate random points lying on a grid.
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

    n_items = len(data)
    partition_size = n_items // n_partitions + 1
    partitions = [
        data[i : i + partition_size] for i in range(0, n_items, partition_size)
    ]
    return partitions


@public
def fill_diagonal(array: Array, val: float | int):
    assert array.ndim >= 2
    i, j = jnp.diag_indices(min(array.shape[-2:]))
    return array.at[..., i, j].set(val)


@public
@partial(jax.jit, static_argnames=["dist_fn", "condensed"])
def dist_matrix(vectors, dist_fn, condensed=False):
    """
    Compute pairwise distances between vectors.

    Examples:
        >>> vectors = jnp.array([[0, 0], [1, 0]])
        >>> dist_fn = lambda x, y: jnp.linalg.norm(x - y)

        >>> dist_matrix(vectors, dist_fn).tolist()
        [[0.0, 1.0], [1.0, 0.0]]

        >>> dist_matrix(vectors, dist_fn, condensed=True).tolist()
        [1.0]
    """
    dists = jax.vmap(jax.vmap(dist_fn, in_axes=(None, 0)), in_axes=(0, None))(
        vectors, vectors
    )

    if condensed:
        size = vectors.shape[0]
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
def dists_to_nearest_neighbor(vectors, dist_fn):
    """Compute pairwise distances between vectors."""
    dists_ = dist_matrix(vectors, dist_fn)
    dists_ = fill_diagonal(dists_, jnp.inf)
    return jnp.min(dists_, axis=0)


@public
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


@public
def fold(
    vec: NDArray,
    dim: int,
    *,
    dtype: NPDTypeLike | None = None,
) -> np.ndarray:
    """
    Reduce vector dimension by folding.

    Examples:
        Fold to a specific size:
        >>> fold([1, 0, 1, 0, 0, 0], dim=3)
        array([1, 0, 1])

        Folding a binary vector returns a binary vector:
        >>> fold([True, False, True, False, False, False], dim=2)
        array([2, 0])

        Specify `dtype` to change the type of the output:
        >>> fold([1, 0, 1, 0, 0, 0], dim=2, dtype=bool)
        array([ True, False])
    """
    vec = np.asarray(vec)
    pad_width = dim - vec.size % dim
    vec = (
        np.pad(vec, (0, pad_width), mode="constant", constant_values=0)
        .reshape(-1, dim)
        .sum(axis=0)
    )
    if dtype is not None:
        vec = vec.astype(dtype)
    return vec


@partial(
    jax.jit,
    static_argnames=["dim", "dtype"],
    backend="cpu",
)
def fold_jax(
    vec: ArrayLike,
    dim: int,
    *,
    dtype: DTypeLike | None = None,
) -> Array:
    """
    Reduce vector dimension by folding.

    Examples:
        Fold to a specific size:
        >>> fold_jax([1, 0, 1, 0, 0, 0], dim=3)
        Array([1, 0, 1], dtype=int32)

        Folding a binary vector returns a binary vector:
        >>> fold_jax([True, False, True, False, False, False], dim=2)
        Array([2, 0], dtype=int32)

        Specify `dtype` to change the type of the output:
        >>> fold_jax([1, 0, 1, 0, 0, 0], dim=2, dtype=bool)
        Array([ True, False], dtype=bool)
    """
    vec = jnp.asarray(vec)
    pad_width = dim - vec.size % dim
    vec = (
        jnp.pad(vec, (0, pad_width), mode="constant", constant_values=0)
        .reshape(-1, dim)
        .sum(axis=0)
    )
    if dtype is not None:
        vec = vec.astype(dtype)
    return vec


@public
def get_named_entity(module: str, name: str):
    """
    Return object from module by name.

    Examples:
        >>> get_named_entity("functools", "partial")
        <class 'functools.partial'>

        >>> get_named_entity("moll.metrics", "does_not_exist")
        Traceback (most recent call last):
            ...
        ValueError: Could not find `does_not_exist` in ...

        >>> get_named_entity("moll.metrics", "euclidean")
        <PjitFunction of ...>
    """
    try:
        module = __import__(module, fromlist=[name])
        return getattr(module, name)
    except (ImportError, AttributeError):
        raise ValueError(f"Could not find `{name}` in `{module}`") from None


@public
def get_function_from_literal(fn: str | Callable, module: str):
    """
    Convert a literal name to a function or return the function as is.
    """
    if isinstance(fn, str):
        return get_named_entity(module, fn)
    return fn
