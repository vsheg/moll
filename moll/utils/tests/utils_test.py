import jax
import jax.numpy as jnp
import pytest

from ..utils import matrix_cross_sum


@pytest.mark.parametrize(
    "matrix, i, j, crossover, expected",
    [
        ([[1, 0], [0, 2]], 0, 0, True, 1),
        ([[1, 0], [0, 2]], 0, 0, False, 0),
        ([[1, 0], [0, 2]], 1, 1, True, 2),
        ([[1, 1], [0, 2]], 0, 1, False, 3),
        ([[1, 2], [3, 4]], 0, 0, True, 6),
        ([[1, 2], [3, 4]], 1, 1, True, 9),
        ([[1, 2], [3, 4]], 1, 1, False, 5),
        ([[1, 3, 4], [3, 2, 5], [4, 5, 3]], 0, 0, True, 15),
    ],
)
def test_matrix_cross_sum(matrix, i, j, crossover, expected):
    matrix = jnp.array(matrix)
    result = matrix_cross_sum(matrix, i, j, crossover=crossover)
    assert result == expected
