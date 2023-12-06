import jax
import jax.numpy as jnp
import pytest

from ...metrics.metrics import euclidean
from ..online_add import (
    _add_point,
    _finalize_updates,
    _needless_point_idx,
    update_points,
)

# Test finalize_updates

finalize_updates = jax.jit(_finalize_updates)


@pytest.mark.parametrize(
    "array, expected",
    [
        ([-1], [-1]),
        ([2], [2]),
        ([-1, -1, -1], [-1, -1, -1]),
        ([3, 3, 3], [-1, -1, 3]),
        ([1, 2, 3], [1, 2, 3]),
        ([1, 1, 2, 2, 3, 3], [-1, 1, -1, 2, -1, 3]),
        ([1, 2, 2, 3, 3, 3], [1, -1, 2, -1, -1, 3]),
        ([3, 2, 1], [3, 2, 1]),
        ([3, 3, -1, 2, 2, -1, 1], [-1, 3, -1, -1, 2, -1, 1]),
    ],
)
def test_finalize_updates(array, expected):
    array = jnp.array(array)
    expected = jnp.array(expected)
    output_array = finalize_updates(array)
    assert jnp.all(output_array == expected)


# Test needless point search

needless_point_idx = jax.jit(
    _needless_point_idx, static_argnames=["dist_fn", "potential_fn"]
)


@pytest.mark.parametrize(
    "array, expected",
    [
        ([0, 0.1, 1], 1),
        ([0, 0, 1], 0),
        ([0, 0, 0, 1], 0),
        ([(0.1, 0.1), (0, 0), (1, 1)], 0),
    ],
)
def test_find_needless_point(array, expected):
    array = jnp.array(array)
    idx = needless_point_idx(array, euclidean, lambda x: x**-2)
    assert idx == expected


# Test add points

add_point = jax.jit(
    _add_point,
    static_argnames=[
        "similarity_fn",
        "potential_fn",
        "k_neighbors",
    ],
)


@pytest.fixture
def X():
    return jnp.array(
        [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
        ],
        dtype=jnp.float32,
    )


def test_add_point(X):
    x = jnp.array([4.3, 4.3])
    X_updated, is_accepted, updated_idx = add_point(
        x=x,
        X=X,
        similarity_fn=euclidean,
        potential_fn=lambda d: d**-1,
        k_neighbors=5,
        n_valid_points=5,
        threshold=0.0,
    )
    assert is_accepted
    assert updated_idx == 4
    assert (X[:4] == X_updated[:4]).all()
    assert (X_updated[4] == x).all()


@pytest.mark.parametrize(
    "xs, acc_mask",
    [
        (
            [[4.3, 4.3], [4.2, 4.2], [4.1, 4.1]],
            [True, False, False],
        ),
        (
            [[4.1, 4.1], [4.2, 4.2], [4.3, 4.3]],
            [False, False, True],
        ),
    ],
)
def test_update_points(X, xs, acc_mask):
    xs = jnp.array(xs)
    acc_mask = jnp.array(acc_mask)
    X_copy = X.copy()

    _updated_idxs, X_updated, acceptance_mask = update_points(
        X=X,
        xs=xs,
        similarity_fn=euclidean,
        potential_fn=lambda d: d**-1,
        k_neighbors=5,
        n_valid_points=5,
        threshold=0.0,
    )

    assert (acceptance_mask == acc_mask).all()
    assert (X_copy[:3] == X_updated[:3]).all()
    idx = jnp.argmax(acc_mask)
    assert (X_updated[4] == xs[idx]).all()
