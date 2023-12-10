"""
Online algorithm for picking a subset of points based on their distance.
"""

from collections.abc import Callable
from typing import List, Literal, Sequence, TypeAlias

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike
from loguru import logger
from numpy.typing import NDArray

from ..metrics import (
    euclidean,
    manhattan,
    mismatches,
    negative_cosine,
    one_minus_tanimoto,
)
from .online_add import update_points

__all__ = ["OnlineDiversityPicker"]

SimilarityFnLiteral = Literal[
    "euclidean", "manhattan", "one_minus_tanimoto", "mismatches", "negative_cosine"
]
SimilarityFn: TypeAlias = Callable[[Array, Array], float] | SimilarityFnLiteral

PotentialFnLiteral = Literal["hyperbolic", "exp", "lj", "log"]
PotentialFn: TypeAlias = Callable[[float], float] | PotentialFnLiteral


class OnlineDiversityPicker:
    """
    Greedy algorithm for picking a diverse subset of points in an online fashion.
    """

    def __init__(
        self,
        capacity: int,
        similarity_fn: SimilarityFn = "euclidean",
        *,
        potential_fn: PotentialFn = "hyperbolic",
        p: float = 1.0,
        k_neighbors: int | float = 5,  # TODO: add heuristic for better default
        threshold: float = -jnp.inf,
        dtype: DTypeLike | None = None,
    ):
        self.capacity = capacity

        self.similarity_fn = self._init_similarity_fn(similarity_fn)

        self.k_neighbors = self._init_k_neighbors(k_neighbors, capacity)

        self.p = p
        self.potential_fn = self._init_potential_fn(potential_fn, self.p)

        self.threshold = threshold

        self._data: Array | None = None
        self.dtype: DTypeLike | None = dtype
        self._labels: NDArray = np.array([None] * capacity, dtype=object)

        self.n_seen: int = 0
        self.n_accepted: int = 0
        self.n_valid_points: int = 0

    def _init_k_neighbors(self, k_neighbors: int | float, capacity: int) -> int:
        if isinstance(k_neighbors, float):
            if not 0 < k_neighbors < 1:
                raise ValueError(f"Float k_neighbors={k_neighbors} must be >0 and <1")

            if k_neighbors * capacity < 1:
                raise ValueError(
                    f"Given the capacity={capacity}, k_neighbors={k_neighbors} is too small",
                )

            k_neighbors = int(k_neighbors * capacity)

        if isinstance(k_neighbors, int):
            if k_neighbors <= 0:
                raise ValueError(f"Expected k_neighbors={k_neighbors} to be positive")
            if k_neighbors > capacity:
                raise ValueError(
                    f"Expected k_neighbors={k_neighbors} to be smaller than capacity={capacity}"
                )

        return k_neighbors

    def _init_similarity_fn(
        self, similarity_fn: SimilarityFn
    ) -> Callable[[Array, Array], float]:
        match similarity_fn:
            case "euclidean":
                return euclidean
            case "manhattan":
                return manhattan
            case "mismatches":
                return mismatches
            case "one_minus_tanimoto":
                return one_minus_tanimoto
            case "negative_cosine":
                return negative_cosine
        return similarity_fn

    def _init_potential_fn(
        self, potential_fn: PotentialFn, p: float
    ) -> Callable[[float], ArrayLike]:
        match potential_fn:
            case "hyperbolic":
                return lambda d: jnp.where(d > 0, jnp.power(d, -p), jnp.inf)
            case "exp":
                return lambda d: jnp.exp(-p * d)
            case "lj":
                sigma: float = p * 0.5 ** (1 / 6)
                return lambda d: (
                    jnp.where(
                        d > 0,
                        jnp.power(sigma / d, 12.0) - jnp.power(sigma / d, 6.0),
                        jnp.inf,
                    )
                )
            case "log":
                return lambda d: jnp.where(d > 0, -jnp.log(p * d), jnp.inf)
        return potential_fn

    def _init_data(self, point: Array, label=None):
        """Initialize the picker with the first point."""
        dim = point.shape[0]
        self.dtype = point.dtype
        self._data = jnp.zeros((self.capacity, dim), dtype=self.dtype)
        self._data = self._data.at[0].set(point)
        self._labels[0] = label

        self.n_valid_points += 1
        self.n_accepted += 1
        self.n_seen += 1

    def _update_labels(
        self,
        updated_idxs: Array,
        acceptance_mask: Array,
        labels: Sequence,  # type: ignore
    ):
        # for updated_idx, label in zip(updated_idxs, labels, strict=True):
        #     if updated_idx >= 0:
        #         self._labels[updated_idx] = label

        idxs = updated_idxs[acceptance_mask]

        # If we create the array with a tuple-like label, we get a 2D array.
        # Besides, an empty 1D array is created explicitly, and then it can be filled
        _labels = labels  # copy value
        labels: NDArray = np.empty(len(labels), dtype=object)  # init 1D array
        labels[:] = _labels  # fill it

        # Update labels
        self._labels[idxs] = labels[acceptance_mask]

    def update(self, points: Array, labels=None) -> int:
        """
        Add a batch of points to the picker.
        """

        batch_size = len(points)
        n_accepted = 0

        # Check labels

        if not labels:
            labels = range(self.n_seen, self.n_seen + len(points))
        else:
            assert len(labels) == batch_size

        # Check dtype

        if points.dtype is jnp.float64:
            points = points.astype(self.dtype)
            logger.warning(
                "Downcasting to float32. If float64 is needed, set dtype=jnp.float64"
                " and configure JAX to support it."
            )

        # Init if empty

        if was_empty := self.is_empty():
            self._init_data(points[0], labels[0])
            points = points[1:]
            labels = labels[1:]
            n_accepted += 1

        # Process remaining points

        if points.shape[0] > 0:
            (
                data_updated,
                updated_idxs,
                acceptance_mask,
                n_appended,
                n_updated,
            ) = update_points(
                X=self._data,
                xs=points,
                similarity_fn=self.similarity_fn,
                potential_fn=self.potential_fn,
                k_neighbors=self.k_neighbors,
                threshold=self.threshold,
                n_valid=self.n_valid_points,
            )

            # Update points data
            self._data = data_updated

            # Update counters
            self.n_valid_points += int(n_appended)
            n_accepted += int(n_appended) + int(n_updated)

            # Update labels
            self._update_labels(updated_idxs, acceptance_mask, labels)

        n_accepted = min(n_accepted, self.capacity)
        self.n_accepted += n_accepted - was_empty
        self.n_seen += batch_size - was_empty

        return n_accepted

    def add(self, point: Array, label=None) -> bool:
        """
        Add a point to the picker.
        """
        points = jnp.array([point])
        labels = [label] if label else None
        n_accepted = self.update(points, labels)
        is_accepted = n_accepted > 0
        return is_accepted

    def warm(self, points: Array, labels=None):
        """
        Initialize the picker with a set of points.
        """
        assert self.is_empty()

        batch_size = points.shape[0]
        assert batch_size <= self.capacity

        if labels:
            assert len(labels) == batch_size
        else:
            labels = range(len(points))

        self._init_data(points[0], labels[0])
        self._data = self._data.at[1:batch_size].set(points[1:])  # type: ignore
        self._labels[1:batch_size] = labels[1:]

        self.n_accepted += batch_size - 1
        self.n_seen += batch_size - 1
        self.n_valid_points += batch_size - 1

    @property
    def n_rejected(self) -> int:
        """
        Return the number of rejected points.
        """
        return self.n_seen - self.n_accepted

    def size(self) -> int:
        """
        Return the number of points in the picker.
        """
        assert self.n_valid_points <= self.capacity
        return self.n_valid_points

    def is_full(self) -> bool:
        """
        Return whether the picker is full or not.
        """
        if self.is_empty():
            return False
        return self.capacity == self.size()

    def is_empty(self) -> bool:
        """
        Return whether the picker is empty or not.
        """
        data = self._data
        return (data is None) or (data.shape[0] == 0)

    @property
    def labels(self) -> list | None:
        """
        Return the currently picked labels.
        """
        if self._data is None:
            return None
        return self._labels[: self.n_valid_points].tolist()

    @property
    def points(self) -> jnp.ndarray | None:
        """
        Return the currently picked points.
        """
        if self._data is None:
            return None
        return self._data[: self.n_valid_points]

    @property
    def dim(self) -> int | None:
        """
        Return the dimension of the points.
        """
        if (data := self._data) is not None:
            return data.shape[1]
        return None
