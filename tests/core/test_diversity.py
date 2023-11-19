from collections import Counter
from random import shuffle

import jax
import jax.numpy as jnp
import pytest
from sklearn import datasets  # type: ignore

from moll.core.distance import euclidean, tanimoto
from moll.core.diversity import OnlineDiversityPicker
from moll.core.utils import generate_points, random_grid_points

RANDOM_SEED = 42


@pytest.fixture
def picker_euclidean():
    return OnlineDiversityPicker(capacity=5, dist_fn=euclidean)


@pytest.fixture
def picker_tanimoto():
    return OnlineDiversityPicker(capacity=5, dist_fn=tanimoto)


# Test that the picker API works as expected


def test_initialization(picker_euclidean):
    assert picker_euclidean.capacity == 5
    assert picker_euclidean.dim is None


def test_append(picker_euclidean):
    p = jnp.array([1, 2, 3])
    picker_euclidean.append(jnp.array(p))
    assert picker_euclidean.n_seen == 1
    assert picker_euclidean.n_accepted == 1
    assert picker_euclidean.points is not None
    assert picker_euclidean.points.shape == (1, 3)
    assert (picker_euclidean.points[0] == p).all()


def test_append_many(picker_euclidean):
    p = jnp.array([1, 2, 3])
    for _ in range(10):
        picker_euclidean.append(jnp.array(p))

    assert picker_euclidean.n_seen == 10
    assert picker_euclidean.is_empty() is False
    assert picker_euclidean.is_full() is False
    assert picker_euclidean.n_accepted == 1
    assert picker_euclidean.points is not None
    assert picker_euclidean.points.shape == (1, 3)
    assert (picker_euclidean.points[0] == p).all()


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
        dist_fn=euclidean,
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
    params=[[3, 3], [10, 1000]],  # smallest and biggest cluster sizes
)
def centers_and_points(centers, request, seed=RANDOM_SEED):
    """
    Generate random points around cluster centers.
    """
    # sizes of smallest and biggest clusters
    smallest, biggest = request.param

    sizes = jnp.linspace(smallest, biggest, num=len(centers), dtype=int)
    return centers, generate_points(centers, sizes=sizes, radius=1, seed=seed)


@pytest.fixture(
    params=[2, 7],  # batch sizes
)
def n_batches(request):
    return request.param


def test_append_many_random(picker, centers_and_points):
    centers, points = centers_and_points

    assert picker.is_empty()

    # If labels are not provided, generate a placeholder for zip

    for point in points:
        picker.append(point)

    selected_points = picker.points

    # Check number of accepted points
    assert picker.n_seen == len(points)
    assert picker.is_empty() is False
    assert picker.is_full() is True
    assert picker.n_accepted >= len(centers)
    assert picker.n_accepted <= len(points)
    assert picker.size() == len(centers)
    assert picker.n_rejected == picker.n_seen - picker.n_accepted

    # Check if at least one point from each cluster is selected
    for center in centers:
        assert jax.vmap(jnp.linalg.norm)(selected_points - center).min() <= jnp.sqrt(
            picker.dim
        )


def test_extend_many_random(picker, centers_and_points, n_batches):
    centers, points = centers_and_points

    assert picker.is_empty() is True

    batches = jnp.array_split(points, n_batches)

    n_accepted_total = 0

    for batch in batches:
        n_accepted = picker.extend(batch)
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


def test_extend_same_points(picker_euclidean: OnlineDiversityPicker):
    center1 = jnp.array([0, 0, 0])
    center2 = jnp.array([0, 10, 0])
    same_centers = [jnp.array([10, 10, 10]) for _ in range(10)]

    points = [center1, center2] + same_centers
    shuffle(points)

    n_accepted = 0

    for point in points:
        n_accepted += picker_euclidean.append(point)

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


def test_labels_append(picker_euclidean: OnlineDiversityPicker, circles):
    points, labels = circles

    for point, label in zip(points, labels):
        picker_euclidean.append(point, label=label)

    assert picker_euclidean.labels
    counts = Counter(circle for circle, idx in picker_euclidean.labels)

    assert counts["large"] >= 4
    assert counts["small"] <= 1


def test_manual_labels_extend(picker_euclidean: OnlineDiversityPicker, circles):
    points, labels = circles

    _n_accepted = picker_euclidean.extend(points, labels=labels)

    assert picker_euclidean.labels
    counts = Counter(circle for circle, idx in picker_euclidean.labels)

    assert counts["large"] >= 4
    assert counts["small"] <= 1


def test_auto_labels_extend(picker_euclidean: OnlineDiversityPicker, circles):
    points, _labels = circles

    large_circle_idxs = {idx for tag, idx in _labels if tag == "large"}
    small_circle_idxs = {idx for tag, idx in _labels if tag == "small"}

    _n_accepted = picker_euclidean.extend(points)

    assert picker_euclidean.labels
    labels_generated = set(picker_euclidean.labels)

    assert large_circle_idxs.issuperset(labels_generated)
    assert small_circle_idxs & labels_generated == set()


def test_tanimoto_picker(picker_tanimoto: OnlineDiversityPicker):
    points = jnp.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
        ]
    )

    picker_tanimoto.extend(points)

    assert picker_tanimoto.n_seen == len(points)
    assert picker_tanimoto.n_accepted == 5


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

    picker_euclidean.fast_init(batch_init, labels_init)
    picker_euclidean.extend(batch, labels)

    assert picker_euclidean.labels
    counts = Counter(circle for circle, idx in picker_euclidean.labels)

    assert counts["large"] >= 4
    assert counts["small"] <= 1
