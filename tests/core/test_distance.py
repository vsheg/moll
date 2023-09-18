import jax.numpy as jnp
from jax import random

from moll.core.distance import euclidean, tanimoto


# Tanimoto tests
def test_tanimoto():
    a = jnp.array([1, 1, 1, 1])
    b = jnp.array([0, 0, 1, 1])
    c = jnp.array([1, 1, 1, 0])
    d = jnp.array([0, 0, 0, 0])
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


# # Distances tests
# def test_distances_single_point():
#     v = jnp.array([1, 2, 3])
#     A = jnp.array([[4, 5, 6]])
#     assert jnp.allclose(distances(v, A), jnp.array([5.196152]))

# def test_distances_multiple_points():
#     key = random.PRNGKey(0)
#     v = random.normal(key, (3,))
#     A = random.normal(key, (10, 3))
#     expected = jnp.sqrt(jnp.sum((A - v) ** 2, axis=-1))
#     assert jnp.allclose(distances(v, A), expected)
