from random import shuffle

import jax
import jax.numpy as jnp
import pytest

from moll.core.distance import euclidean
from moll.core.diversity import OnlineDiversityPicker


@pytest.fixture
def picker5():
    return OnlineDiversityPicker(n_points=5, dist=euclidean)


@pytest.fixture
def picker10():
    return OnlineDiversityPicker(n_points=10, dist=euclidean)


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


def points_around(center, n_points=10, radius=1, dim=3):
    key = jax.random.PRNGKey(0)
    points = []
    for _ in range(n_points):
        key, subkey = jax.random.split(key)
        offset = jax.random.uniform(subkey, shape=(dim,), minval=-radius, maxval=radius)
        points.append(center + offset)
    return jnp.array(points)


def test_append_many_random_points(picker5: OnlineDiversityPicker):
    center1 = jnp.array([0, 0, 0])
    center2 = jnp.array([0, 10, 0])
    center3 = jnp.array([10, 10, 10])

    cluster1 = points_around(center1, n_points=1)
    cluster2 = points_around(center2, n_points=10)
    cluster3 = points_around(center3, n_points=100)

    points = jnp.concatenate([cluster1, cluster2, cluster3], axis=0)
    key = jax.random.PRNGKey(0)
    points = jax.random.permutation(key, points)

    for point in points:
        picker5.append(point)

    selected_points = picker5.points

    # Check number of accepted points
    assert picker5.n_seen == 111
    assert picker5.is_empty() == False
    assert picker5.is_full() == True
    assert picker5.n_accepted >= 5
    assert picker5.n_rejected == picker5.n_seen - picker5.n_accepted

    # Check if at least one point from each cluster is in the selected points
    assert jnp.isin(cluster1, selected_points).any()
    assert jnp.isin(cluster2, selected_points).any()
    assert jnp.isin(cluster3, selected_points).any()


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


def test_extend_random_points(picker10: OnlineDiversityPicker):
    assert picker10.is_empty() == True

    n_clusters = 4

    center1 = jnp.array([0] * 10)
    center2 = jnp.array([0, 10] * 5)
    center3 = jnp.array([10, 0] * 5)
    center4 = jnp.array([10] * 10)

    cluster1 = points_around(center1, dim=10, n_points=1)
    cluster2 = points_around(center2, dim=10, n_points=10)
    cluster3 = points_around(center3, dim=10, n_points=100)
    cluster4 = points_around(center4, dim=10, n_points=1000)

    points = jnp.concatenate([cluster1, cluster2, cluster3, cluster4], axis=0)
    key = jax.random.PRNGKey(1)
    points = jax.random.permutation(key, points)

    batches = jnp.array_split(points, 10)

    n_accepted_total = 0

    for batch in batches:
        n_accepted = picker10.extend(batch)
        assert n_accepted >= 0
        assert n_accepted <= picker10.n_points
        n_accepted_total += n_accepted

    selected_points = picker10.points

    # Check if at least one point from each cluster is in the selected points
    assert picker10.n_accepted >= n_clusters  # n_clusters
    assert picker10.n_accepted == n_accepted_total
    assert picker10.n_seen == 1111
    assert picker10.n_rejected == picker10.n_seen - picker10.n_accepted
    assert jnp.isin(cluster1, selected_points).any()
    assert jnp.isin(cluster2, selected_points).any()
    assert jnp.isin(cluster3, selected_points).any()
    assert jnp.isin(cluster4, selected_points).any()
