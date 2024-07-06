"""
Type aliases and protocols used throughout the package.
"""
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from jax import Array
from jax.typing import ArrayLike
from rdkit import Chem

__all__ = [
    "DistanceFnLiteral",
    "DistanceFnCallable",
    "SimilarityFnLiteral",
    "SimilarityFnCallable",
    "LossFnLiteral",
    "LossFnCallable",
    "Indexable",
    "OneOrMany",
]

DistanceFnLiteral = Literal[
    "euclidean", "manhattan", "one_minus_tanimoto", "mismatches", "negative_cosine"
]
DistanceFnCallable: TypeAlias = Callable[[Array, Array], ArrayLike]

SimilarityFnLiteral = Literal["identity"]
SimilarityFnCallable: TypeAlias = Callable[[Array], ArrayLike]

# TODO: maybe use abbreviations?
LossFnLiteral = Literal[
    "power",
    "exponential",
    "lennard_jones",
    "logarithmic",
]
LossFnCallable: TypeAlias = Callable[[float], ArrayLike]


RDKitMol: TypeAlias = Chem.rdchem.Mol
RDKitAtom: TypeAlias = Chem.rdchem.Atom
SMILES: TypeAlias = str

FingerprintLiteral = Literal["morgan"]

PathLike: TypeAlias = str | Path

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
    TYPE_CHECKING,
    Path,
    Chem,
)
