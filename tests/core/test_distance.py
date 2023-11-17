import jax.numpy as jnp
import pytest

from moll.core.distance import (
    add_point_to_bag,
    add_points_to_bag,
    euclidean,
    finalize_updates,
    matrix_cross_sum,
    needless_point_idx,
    tanimoto,
)


# Tanimoto tests
@pytest.mark.parametrize(
    "dtype",
    [
        bool,
        jnp.bool_,
        int,
        jnp.int8,
        jnp.int16,
        jnp.uint8,
        jnp.uint16,
    ],
)
def test_tanimoto(dtype):
    a = jnp.array([1, 1, 1, 1]).astype(dtype)
    b = jnp.array([0, 0, 1, 1]).astype(dtype)
    c = jnp.array([1, 1, 1, 0]).astype(dtype)
    d = jnp.array([0, 0, 0, 0]).astype(dtype)

    assert tanimoto(a, b) == 0.5
    assert tanimoto(a, c) == 0.25
    assert tanimoto(a, d) == 1.0
    assert tanimoto(b, c) == 0.75
    assert tanimoto(b, d) == 1.0
    assert tanimoto(a, a) == 0.0
    assert tanimoto(d, d) == 0.0


# Euclidean tests
def test_euclidean():
    p1 = jnp.array([1, 2, 3])
    p2 = jnp.array([4, 5, 6])
    expected_result = jnp.sqrt(27)
    assert euclidean(p1, p2) == expected_result


# Test finalize_updates


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


def dummy_dist_fn(x, y):
    return jnp.linalg.norm(x - y)


def dummy_potential_fn(dist):
    return dist**2


# Test needless point search


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


@pytest.mark.parametrize(
    "matrix, i, j, expected",
    [
        ([[1, 0], [0, 2]], 0, 0, 1),
        ([[1, 0], [0, 2]], 1, 1, 2),
        ([[1, 2], [3, 4]], 0, 0, 6),
        ([[1, 2], [3, 4]], 1, 1, 9),
        ([[1, 3, 4], [3, 2, 5], [4, 5, 3]], 0, 0, 15),
    ],
)
def test_matrix_cross_sum(matrix, i, j, expected):
    matrix = jnp.array(matrix)
    result = matrix_cross_sum(matrix, i, j)
    assert result == expected


# Test add points


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


def test_add_point_to_bag(X):
    x = jnp.array([4.3, 4.3])
    X_updated, is_accepted, updated_idx = add_point_to_bag(
        x=x,
        X=X,
        dist_fn=euclidean,
        k_neighbors=5,
        power=2,
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
def test_add_points_to_bag(X, xs, acc_mask):
    xs = jnp.array(xs)
    acc_mask = jnp.array(acc_mask)

    update_idxs, X_updated, acceptance_mask = add_points_to_bag(
        X=X,
        xs=xs,
        dist_fn=euclidean,
        k_neighbors=5,
        power=2,
        n_valid_points=5,
        threshold=0.0,
    )

    assert (acceptance_mask == acc_mask).all()
    assert (X[:3] == X_updated[:3]).all()
    idx = jnp.argmax(acc_mask)
    assert (X_updated[4] == xs[idx]).all()
