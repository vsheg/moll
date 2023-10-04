import itertools
import uuid
from random import sample, shuffle
from time import time

import jax
import jax.numpy as jnp
import pytest

from moll.core.distance import euclidean
from moll.core.diversity import OnlineDiversityPicker
from moll.core.utils import generate_points, partition


@pytest.fixture
def picker5():
    return OnlineDiversityPicker(n_points=5, dist=euclidean)


# Test that the picker API works as expected


def test_initialization(picker5: OnlineDiversityPicker):
    assert picker5.n_points == 5
    assert picker5.dim is None


def test_append(picker5: OnlineDiversityPicker):
    v = [1, 2, 3]
    picker5.append(jnp.array(v))
    assert picker5.n_seen == 1
    assert picker5.n_accepted == 1
    assert picker5.data_.shape == (1, 3)
    assert picker5.data_[0].tolist() == v


def test_append_many(picker5: OnlineDiversityPicker):
    v = [1, 2, 3]
    for _ in range(10):
        picker5.append(jnp.array(v))
    assert picker5.n_seen == 10
    assert picker5.is_empty() == False
    assert picker5.is_full() == False
    assert picker5.n_accepted == 1
    assert picker5.data_.shape == (1, 3)
    assert picker5.data_[0].tolist() == v


# Test that the picker returns the most distant points


@pytest.fixture(
    params=[3, 5],  # picker sizes
)
def picker(request):
    """
    Return a picker with a different number of points.
    """
    return OnlineDiversityPicker(n_points=request.param, dist=euclidean)


@pytest.fixture(
    params=[2, 3],  # dimentions
)
def centers(picker, request):
    """
    Generate random cluster centers.
    """
    n_centers = picker.n_points
    dim = request.param

    # generate all possible centers in the grid of size = 2 * `dim` + 1
    grid = range(-dim, dim + 1)
    grid_points = itertools.product(grid, repeat=dim)
    random_grid_points = sample(list(grid_points), k=n_centers)
    centers = jnp.array(random_grid_points)

    # TODO: do not generate all possible grid points, just sample

    return centers


@pytest.fixture(
    params=[[10, 10], [1, 1000]],  # smallest and biggest cluster sizes
)
def points(centers, request, seed=42):
    """
    Generate random points around cluster centers.
    """
    # sizes of smallest and biggest clusters
    smallest, biggest = request.param

    sizes = jnp.linspace(smallest, biggest, num=len(centers), dtype=int)
    return generate_points(centers, sizes=sizes, radius=1, seed=seed)


@pytest.fixture(
    params=[True, False],  # whether to generate labels or not
)
def labels(points, request):
    """
    Generate random labels for points.
    """
    if request.param:
        return [str(uuid.uuid4()) for _ in range(len(points))]
    return None


@pytest.fixture(
    params=[2, 9],  # batch sizes
)
def n_batches(request):
    return request.param


def test_append(picker, points, labels, centers):
    assert picker.is_empty() == True

    # If labels are not provided, generate a placeholder for zip
    labels_ = labels or itertools.repeat(None)

    for point, label in zip(points, labels_):
        if labels:
            picker.append(point, label=label)
        else:
            picker.append(point)

    selected_points = picker.points

    # Check number of accepted points
    assert picker.n_seen == len(points)
    assert picker.is_empty() == False
    assert picker.is_full() == True
    assert picker.n_accepted >= len(centers)
    assert picker.n_accepted <= len(points)
    assert picker.size() == len(centers)
    assert picker.n_rejected == picker.n_seen - picker.n_accepted

    # Check labels
    if labels:
        assert len(picker.labels) == picker.n_points
        assert len(set(picker.labels)) == picker.n_points
        assert set(picker.labels).issubset(set(labels))

    # Check if at least one point from each cluster is selected
    for center in centers:
        assert (jnp.abs(center - selected_points) <= 1).any()


def test_extend(picker, centers, points, labels, n_batches):
    assert picker.is_empty() == True

    batches = jnp.array_split(points, n_batches)

    n_accepted_total = 0
    next_label_to_take_idx = 0

    for batch in batches:
        if labels:
            batch_size = len(batch)
            print("next_label_to_take_idx:", next_label_to_take_idx)
            label_batch = labels[
                next_label_to_take_idx : next_label_to_take_idx + batch_size
            ]
            next_label_to_take_idx += batch_size
            print("labels:", labels)
            print("label_batch", label_batch)
            print("picker.labels", picker.labels)
            n_accepted = picker.extend(batch, labels=label_batch)
            print("here!")
        else:
            n_accepted = picker.extend(batch)

        assert n_accepted >= 0
        assert n_accepted <= picker.n_points
        n_accepted_total += n_accepted

    selected_points = picker.points

    # Check if at least one point from each cluster is in the selected points
    assert picker.n_accepted >= len(centers)
    assert picker.n_accepted == n_accepted_total
    assert picker.n_seen == len(points)
    assert picker.n_rejected == picker.n_seen - picker.n_accepted

    # Check if at least one point from each cluster is selected
    for center in centers:
        assert (jnp.abs(center - selected_points) <= 1).any()


def test_extend_same_points(picker5: OnlineDiversityPicker):
    center1 = jnp.array([0, 0, 0])
    center2 = jnp.array([0, 10, 0])
    same_centers = [jnp.array([10, 10, 10]) for _ in range(10)]

    points = [center1, center2] + same_centers
    shuffle(points)

    n_accepted = 0

    for point in points:
        n_accepted += picker5.append(point)

    selected_points = picker5.points

    # Check number of accepted points
    assert picker5.n_seen == 12
    assert picker5.n_accepted == n_accepted
    assert picker5.n_accepted == 3
    assert picker5.n_rejected == 9
    assert picker5.points.shape == (3, 3)
    assert len(picker5.labels) == n_accepted
