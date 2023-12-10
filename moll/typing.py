from collections.abc import Callable
from typing import Literal, Protocol, TypeAlias, TypeVar

from jax import Array
from jax.typing import ArrayLike

SimilarityFnLiteral = Literal[
    "euclidean", "manhattan", "one_minus_tanimoto", "mismatches", "negative_cosine"
]
SimilarityFnCallable: TypeAlias = Callable[[Array, Array], ArrayLike]

PotentialFnLiteral = Literal["hyperbolic", "exp", "lj", "log"]
PotentialFnCallable: TypeAlias = Callable[[float], ArrayLike]

T = TypeVar("T", covariant=True)


class Indexable(Protocol[T]):
    def __getitem__(self, key) -> T:
        ...

    def __len__(self) -> int:
        ...
