import jax.numpy as jnp
import pytest

from ...metrics import euclidean
from .._online_add import (
    _add_vector,
    _finalize_updates,
    _needless_vector_idx,
    update_vectors,
)

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
    output_array = _finalize_updates(array)
    assert jnp.all(output_array == expected)


# Test needless vector search


@pytest.mark.parametrize(
    "array, expected",
    [
        ([0, 0.1, 1], 1),
        ([0, 0, 1], 0),
        ([0, 0, 0, 1], 0),
        ([(0.1, 0.1), (0, 0), (1, 1)], 0),
    ],
)
@pytest.mark.parametrize(
    "dist_fn",
    [
        euclidean,
        lambda x, y: euclidean(x, y) - 10,  # negative distance is ok
    ],
)
def test_find_needless_vector(array, expected, dist_fn):
    array = jnp.array(array)
    # exp potential is used to treat negative distances
    idx = _needless_vector_idx(array, dist_fn, lambda d: jnp.exp(-d))
    assert idx == expected


# Test add vectors


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


@pytest.mark.parametrize(
    "similarity_fn",
    [
        euclidean,
        lambda x, y: euclidean(x, y) + 10,
        lambda x, y: euclidean(x, y) - 10,
    ],
)
def test_add_vector(X, similarity_fn):
    x = jnp.array([4.3, 4.3])
    X_copy = X.copy()
    X_updated, updated_idx = _add_vector(
        x=x,
        X=X,
        similarity_fn=similarity_fn,
        potential_fn=lambda d: jnp.exp(-d),
        k_neighbors=5,
        n_valid_vectors=5,
        threshold=-jnp.inf,
    )
    assert updated_idx >= 0
    assert updated_idx == 4
    assert (X_copy[:4] == X_updated[:4]).all()
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
def test_update_vectors(X, xs, acc_mask):
    xs = jnp.array(xs)
    acc_mask = jnp.array(acc_mask)
    X_copy = X.copy()

    X_updated, updated_idxs, acceptance_mask, n_appended, n_updated = update_vectors(
        X=X,
        xs=xs,
        similarity_fn=euclidean,
        potential_fn=lambda d: d**-1,
        k_neighbors=5,
        n_valid=5,
        threshold=0.0,
    )

    assert (acceptance_mask == acc_mask).all()
    assert (X_copy[:3] == X_updated[:3]).all()
    idx = jnp.argmax(acc_mask)
    assert (X_updated[4] == xs[idx]).all()
