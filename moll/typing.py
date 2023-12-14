"""
Type aliases and protocols used throughout the package.
"""
from collections.abc import Callable, Iterable
from typing import Literal, Protocol, TypeAlias, TypeVar, runtime_checkable

from jax import Array
from jax.typing import ArrayLike

__all__ = [
    "SimilarityFnLiteral",
    "SimilarityFnCallable",
    "PotentialFnLiteral",
    "PotentialFnCallable",
    "Indexable",
    "OneOrMany",
]

SimilarityFnLiteral = Literal[
    "euclidean", "manhattan", "one_minus_tanimoto", "mismatches", "negative_cosine"
]
SimilarityFnCallable: TypeAlias = Callable[[Array, Array], ArrayLike]

PotentialFnLiteral = Literal["hyperbolic", "exp", "lj", "log"]
PotentialFnCallable: TypeAlias = Callable[[float], ArrayLike]

T = TypeVar("T", covariant=True)


@runtime_checkable
class Indexable(Protocol[T]):
    """
    Protocol for objects that can be accessed by index or slice.
    """

    def __getitem__(self, key) -> T:
        ...

    def __len__(self) -> int:
        ...


del T

T = TypeVar("T")

OneOrMany: TypeAlias = T | Iterable[T]

del T

# Remove third-party imports from the public API
del (
    Callable,
    Iterable,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    Array,
    ArrayLike,
    runtime_checkable,
)
