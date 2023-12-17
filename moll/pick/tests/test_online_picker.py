from collections import Counter
from collections.abc import Sequence
from random import shuffle
from typing import cast, get_args

import jax
import jax.numpy as jnp
import pytest
from sklearn import datasets

from ...metrics import euclidean
from ...utils import dists_to_nearest_neighbor, globs, random_grid_points
from .._online_picker import (
    OnlineDiversityPicker,
    PotentialFnLiteral,
    SimilarityFnLiteral,
)

RANDOM_SEED = 42


@pytest.fixture
def picker_euclidean():
    return OnlineDiversityPicker(capacity=5, similarity_fn=euclidean)


# Test that the picker API works as expected


def test_initialization(picker_euclidean):
    assert picker_euclidean.capacity == 5
    assert picker_euclidean.dim is None


def test_add(picker_euclidean):
    p = jnp.array([1, 2, 3])
    picker_euclidean.add(jnp.array(p))
    assert picker_euclidean.n_seen == 1
    assert picker_euclidean.n_accepted == 1
    assert picker_euclidean.points is not None
    assert picker_euclidean.points.shape == (1, 3)
    assert (picker_euclidean.points[0] == p).all()


def test_add_many(picker_euclidean):
    p = jnp.array([1, 2, 3])
    for _ in range(10):
        picker_euclidean.add(jnp.array(p))

    assert picker_euclidean.n_seen == 10
    assert picker_euclidean.is_empty() is False
    assert picker_euclidean.is_full() is False
    assert picker_euclidean.n_accepted == 1
    assert picker_euclidean.points is not None
    assert picker_euclidean.points.shape == (1, 3)
    assert (picker_euclidean.points[0] == p).all()


@pytest.mark.parametrize(
    "k_neighbors,expected,text",
    [
        (1, 1, ...),
        (5, 5, ...),
        (0.5, 2, ...),  # 0.5 * capacity = 2.5 -> 2
        (0, ValueError, "positive"),
        (-1, ValueError, "positive"),
        (1.5, ValueError, ">0 and <1"),
        (10, ValueError, "smaller"),  # k_neighbors > capacity
    ],
)
def test_k_neighbors(k_neighbors, expected, text):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected, match=text):
            print(k_neighbors)
            picker = OnlineDiversityPicker(
                capacity=5, similarity_fn=euclidean, k_neighbors=k_neighbors
            )
    else:
        picker = OnlineDiversityPicker(
            capacity=5, similarity_fn=euclidean, k_neighbors=k_neighbors
        )
        assert picker.k_neighbors == expected


# Test that the picker returns the most distant points


@pytest.fixture(
    params=[3, 6, 13],  # picker sizes
)
def picker(request):
    """
    Return a picker with a different number of points.
    """
    return OnlineDiversityPicker(
        capacity=request.param,
        similarity_fn=euclidean,
        k_neighbors=request.param,
    )


@pytest.fixture(
    params=[2, 3],  # dimentions
)
def centers(picker, request, seed=RANDOM_SEED, n_ticks=5):
    """
    Generate random cluster centers.
    """
    n_centers = picker.capacity
    dim = request.param

    centers = random_grid_points(
        n_points=n_centers,
        dim=dim,
        n_ticks=n_ticks,
        seed=seed,
        spacing=10,
    )

    return centers


@pytest.fixture(
    params=[[3, 3], [1, 100]],  # smallest and biggest cluster sizes
)
def centers_and_points(centers, request, seed=RANDOM_SEED):
    """
    Generate random points around cluster centers.
    """
    # sizes of smallest and biggest clusters
    smallest, biggest = request.param

    sizes = jnp.linspace(smallest, biggest, num=len(centers), dtype=int)
    sizes = cast(Sequence[int], sizes)
    return centers, globs(centers, sizes=sizes, stds=0.5, cap_radius=1, seed=seed)


@pytest.fixture(
    params=[2, 7],  # batch sizes
)
def n_batches(request):
    return request.param


def test_add_many_random(picker, centers_and_points):
    centers, points = centers_and_points

    assert picker.is_empty()

    # If labels are not provided, generate a placeholder for zip

    for point in points:
        picker.add(point)

    selected_points = picker.points

    # Check number of accepted points
    assert picker.n_seen == len(points)
    assert picker.is_empty() is False
    assert picker.is_full() is True
    assert picker.n_accepted >= len(centers)
    assert picker.n_accepted <= len(points)
    assert picker.size == len(centers)
    assert picker.n_rejected == picker.n_seen - picker.n_accepted

    # Check if at least one point from each cluster is selected
    for center in centers:
        assert jax.vmap(jnp.linalg.norm)(selected_points - center).min() <= jnp.sqrt(
            picker.dim
        )


def test_update_many_random(picker, centers_and_points, n_batches):
    centers, points = centers_and_points

    assert picker.is_empty() is True

    batches = jnp.array_split(points, n_batches)

    n_accepted_total = 0

    for batch in batches:
        n_accepted = picker.update(batch)
        assert n_accepted >= 0
        assert n_accepted <= picker.capacity
        n_accepted_total += n_accepted

    selected_points = picker.points

    # Check if at least one point from each cluster is in the selected points
    assert picker.n_accepted >= len(centers)
    assert picker.n_accepted == n_accepted_total
    assert picker.n_seen == len(points)
    assert picker.n_rejected == picker.n_seen - picker.n_accepted

    # Check if at least one point from each cluster is selected
    for center in centers:
        assert jax.vmap(jnp.linalg.norm)(selected_points - center).min() <= jnp.sqrt(
            picker.dim
        )


def test_update_same_points(picker_euclidean: OnlineDiversityPicker):
    center1 = jnp.array([0, 0, 0])
    center2 = jnp.array([0, 10, 0])
    same_centers = [jnp.array([10, 10, 10]) for _ in range(10)]

    points = [center1, center2] + same_centers
    shuffle(points)

    n_accepted = 0

    for point in points:
        n_accepted += picker_euclidean.add(point)

    _selected_points = picker_euclidean.points

    # Check number of accepted points
    assert picker_euclidean.n_seen == 12
    assert picker_euclidean.n_accepted == n_accepted
    assert picker_euclidean.n_accepted == 3
    assert picker_euclidean.n_rejected == 9

    assert picker_euclidean.points is not None
    assert picker_euclidean.points.shape == (3, 3)

    assert picker_euclidean.labels is not None
    assert len(picker_euclidean.labels) == n_accepted


@pytest.fixture
def circles(factor=0.1, random_state=42, n_samples=20):
    """
    Generate 2 circles: a small one and a large one. Points in the small circle are not
    favorable because of repulsion.
    """
    points, tags = datasets.make_circles(
        factor=factor, random_state=random_state, n_samples=n_samples
    )
    labels = [("small" if tag else "large", i) for i, tag in enumerate(tags)]
    return points, labels


def test_labels_add(picker_euclidean: OnlineDiversityPicker, circles):
    points, labels = circles

    for point, label in zip(points, labels, strict=True):
        picker_euclidean.add(point, label=label)

    assert picker_euclidean.labels
    counts = Counter(circle for circle, idx in picker_euclidean.labels)

    assert counts["large"] >= 4
    assert counts["small"] <= 1


def test_manual_labels_update(picker_euclidean: OnlineDiversityPicker, circles):
    points, labels = circles

    _n_accepted = picker_euclidean.update(points, labels=labels)

    assert picker_euclidean.labels
    counts = Counter(circle for circle, idx in picker_euclidean.labels)

    assert counts["large"] >= 4
    assert counts["small"] <= 1


def test_auto_labels_update(picker_euclidean: OnlineDiversityPicker, circles):
    points, _labels = circles

    large_circle_idxs = {idx for tag, idx in _labels if tag == "large"}
    small_circle_idxs = {idx for tag, idx in _labels if tag == "small"}

    _n_accepted = picker_euclidean.update(points)

    assert picker_euclidean.labels
    labels_generated = set(picker_euclidean.labels)

    assert large_circle_idxs.issuperset(labels_generated)
    assert small_circle_idxs & labels_generated == set()


# Test warm start


@pytest.mark.parametrize(
    "init_batch_size",
    [1, 2, 3, 4, 5],
)
def test_fast_init(picker_euclidean: OnlineDiversityPicker, circles, init_batch_size):
    points, labels = circles

    batch_init = points[:init_batch_size]
    batch = points[init_batch_size:]
    labels_init = labels[:init_batch_size]
    labels = labels[init_batch_size:]

    picker_euclidean.warm(batch_init, labels_init)
    picker_euclidean.update(batch, labels)

    assert picker_euclidean.labels
    counts = Counter(circle for circle, idx in picker_euclidean.labels)

    assert counts["large"] >= 4
    assert counts["small"] <= 1


# Test picker custom similarity functions

similarity_fns: tuple = get_args(SimilarityFnLiteral) + (
    lambda x, y: euclidean(x, y) + 10,  # similarities must me ordered, shift is ok
    lambda x, y: euclidean(x, y) - 10,  # similarities must me ordered, negative is ok
)


@pytest.fixture(params=similarity_fns)
def picker_similarity_fn(request):
    similarity_fn = request.param
    return OnlineDiversityPicker(
        capacity=5,
        similarity_fn=similarity_fn,
        potential_fn="exp",  # exp potential is used to treat negative similarities
    )


@pytest.fixture
def integer_vectors(n_points=1_000, dim=10, seed: int = RANDOM_SEED):
    return jax.random.randint(
        jax.random.PRNGKey(seed),
        shape=(n_points, dim),
        minval=-2,
        maxval=2,
    )


def test_custom_similarity_fn(picker_similarity_fn, integer_vectors):
    picker_similarity_fn.update(integer_vectors)

    assert picker_similarity_fn.n_seen == len(integer_vectors)
    assert picker_similarity_fn.n_accepted == 5

    min_dist_orig = dists_to_nearest_neighbor(integer_vectors, euclidean).min()
    min_dist_new = dists_to_nearest_neighbor(
        picker_similarity_fn.points, euclidean
    ).min()

    # Check that the min pairwise distance is increased by at least a factor:
    factor = 1.5

    assert (min_dist_new > factor * min_dist_orig).all()


# Test custom potential functions


potential_fns: tuple = get_args(PotentialFnLiteral) + (
    lambda d: jnp.exp(d),  # potentials must me ordered, negative is ok
)


@pytest.fixture(params=potential_fns)
def picker_potential_fn(request):
    potential_fn = request.param
    return OnlineDiversityPicker(capacity=5, potential_fn=potential_fn)


@pytest.fixture
def uniform_rectangle(n_points=1_000, dim=2, seed: int = RANDOM_SEED):
    return jax.random.uniform(jax.random.PRNGKey(RANDOM_SEED), (n_points, dim))


def test_custom_potential_fn(picker_potential_fn, uniform_rectangle):
    picker = picker_potential_fn
    picker.update(uniform_rectangle)

    min_dist_orig = dists_to_nearest_neighbor(uniform_rectangle, euclidean).min()
    min_dist_new = dists_to_nearest_neighbor(
        picker_potential_fn.points, euclidean
    ).min()

    # Check that the min pairwise distance is increased by at least a factor:
    factor = 1.5

    assert (min_dist_new > factor * min_dist_orig).all()
