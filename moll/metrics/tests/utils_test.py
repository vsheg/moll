import jax
import jax.numpy as jnp
import pytest

from ..utils import _matrix_cross_sum

matrix_cross_sum = jax.jit(_matrix_cross_sum, static_argnames=["row_only"])


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
