import jax.numpy as jnp
import pytest

from .._metrics import tanimoto


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
    assert tanimoto(a, c) == 0.75
    assert tanimoto(a, d) == 0.0
    assert tanimoto(b, c) == 0.25
    assert tanimoto(b, d) == 0.0
    assert tanimoto(a, a) == 1.0
    assert tanimoto(d, d) == 0.0  # all-zero vectors are interpreted as different
