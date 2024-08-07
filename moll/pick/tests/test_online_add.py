import jax.numpy as jnp
import pytest

from ...measures import euclidean
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
    idx = _needless_vector_idx(
        array, dist_fn, sim_fn=lambda d: d, loss_fn=lambda s: jnp.exp(-s)
    )
    assert idx == expected


# Test needless vector search with maximization


@pytest.mark.parametrize(
    "array, idx_expected",
    [
        ([0, 0.1, 1], 2),
        ([0, 1.1, 1], 0),
        ([0, 0, 0, 1], 3),
        ([(0.1, 0.1), (0, 0), (1, 1)], 2),
    ],
)
@pytest.mark.parametrize(
    "dist_fn",
    [
        euclidean,
        lambda x, y: euclidean(x, y) - 10,  # negative distance is ok
    ],
)
def test_find_needless_vector_with_maximize(array, idx_expected, dist_fn):
    array = jnp.array(array)
    # exp potential is used to treat negative distances
    idx = _needless_vector_idx(
        array, dist_fn, sim_fn=lambda d: d, loss_fn=lambda s: jnp.exp(-s), maximize=True
    )
    assert idx == idx_expected


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
    "dist_fn",
    [
        euclidean,
        lambda x, y: euclidean(x, y) + 10,
        lambda x, y: euclidean(x, y) - 10,
    ],
)
def test_add_vector(X, dist_fn):
    x = jnp.array([4.3, 4.3])
    X_copy = X.copy()
    X_updated, updated_idx = _add_vector(
        x=x,
        X=X,
        dist_fn=dist_fn,
        sim_fn=lambda d: d,
        loss_fn=lambda s: jnp.exp(-s),
        k_neighbors=5,
        n_valid_vectors=5,
        sim_min=-jnp.inf,
    )
    assert updated_idx >= 0
    assert updated_idx == 4
    assert (X_copy[:4] == X_updated[:4]).all()
    assert (X_updated[4] == x).all()


@pytest.mark.parametrize(
    "X_pinned, x, updated_idx",
    [
        (
            [[-100, -100], [-200, -200]],
            [4.3, 4.3],
            4,
        ),
        (
            [[4.31, 4.31]],
            [4.3, 4.3],
            -1,
        ),
    ],
)
def test_add_vector_with_pinned(X, X_pinned, x, updated_idx):
    X_copy = X.copy()
    X_pinned = jnp.array(X_pinned)
    x = jnp.array(x)

    X_updated, upd_idx = _add_vector(
        x=x,
        X=X,
        X_pinned=X_pinned,
        dist_fn=euclidean,
        sim_fn=lambda d: d,
        loss_fn=lambda s: s**-1,
        k_neighbors=5,
        n_valid_vectors=5,
    )
    assert upd_idx == updated_idx

    if updated_idx == -1:
        assert (X_copy == X_updated).all()
    else:
        assert (X_updated == X_copy.at[updated_idx].set(x)).all()


@pytest.mark.parametrize(
    "x, updated_idx",
    [
        ([1.1, 1.1], 4),
        ([0.1, 0.1], 4),
        ([4.1, 4.1], 0),
    ],
)
def test_add_vector_with_maximize(X, x, updated_idx):
    X_copy = X.copy()
    x = jnp.array(x)

    X_updated, upd_idx = _add_vector(
        x=x,
        X=X,
        dist_fn=euclidean,
        sim_fn=lambda d: d,
        loss_fn=lambda s: s**-1,
        k_neighbors=5,
        n_valid_vectors=len(X),
        maximize=True,
    )
    assert upd_idx == updated_idx

    if updated_idx == -1:
        assert (X_copy == X_updated).all()
    else:
        assert (X_updated == X_copy.at[updated_idx].set(x)).all()


@pytest.mark.parametrize(
    "X_pinned, xs, acc_mask",
    [
        (
            None,
            [[4.3, 4.3], [4.2, 4.2], [4.1, 4.1]],
            [True, False, False],
        ),
        (
            None,
            [[4.1, 4.1], [4.2, 4.2], [4.3, 4.3]],
            [False, False, True],
        ),
        (
            [[0.1, 0.1]],
            [[4.3, 4.3], [4.2, 4.2], [4.1, 4.1]],
            [True, False, False],
        ),
        (
            [[4.31, 4.31], [4.21, 4.21], [4.11, 4.11]],
            [[4.3, 4.3], [4.2, 4.2], [4.1, 4.1]],
            [False, False, False],
        ),
        (
            [[6, 6]],
            [[4.3, 4.3], [4.5, 4.5], [4.1, 4.1]],  # (3, 3) ... (?, ?) ... (6, 6)
            [False, True, False],
        ),
    ],
)
def test_update_vectors(X, X_pinned, xs, acc_mask):
    X_pinned = jnp.array(X_pinned) if X_pinned is not None else None
    xs = jnp.array(xs)
    acc_mask = jnp.array(acc_mask)
    X_copy = X.copy()

    X_updated, updated_idxs, acceptance_mask, n_appended, n_updated = update_vectors(
        X=X,
        X_pinned=X_pinned,
        xs=xs,
        dist_fn=euclidean,
        sim_fn=lambda d: d,
        loss_fn=lambda s: s**-1,
        k_neighbors=5,
        n_valid=5,
        sim_min=0.0,
        sim_max=jnp.inf,
    )

    assert X_copy.shape == X_updated.shape
    assert (acceptance_mask == acc_mask).all()
    assert (X_copy[:3] == X_updated[:3]).all()

    any_accepted = jnp.any(acceptance_mask)

    if any_accepted:
        idx = jnp.argmax(acceptance_mask)
        assert (X_updated[4] == xs[idx]).all()
    else:
        assert (X_updated == X_copy).all()
