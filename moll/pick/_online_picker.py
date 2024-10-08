"""
Online algorithm for picking a subset of vectors based on their similarity.
"""


from collections.abc import Hashable, Iterable
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike
from loguru import logger
from numpy.typing import NDArray
from public import public
from sklearn.base import BaseEstimator, TransformerMixin

from ..typing import (
    DistanceFnCallable,
    DistanceFnLiteral,
    Indexable,
    LossFnCallable,
    LossFnLiteral,
    SimilarityFnCallable,
    SimilarityFnLiteral,
)
from ..utils import get_function_from_literal, hasarg
from ._online_add import update_vectors


@public
class OnlineVectorPicker(BaseEstimator, TransformerMixin):
    """
    Greedy algorithm for picking a subset of vectors in an online fashion.
    """

    def __init__(
        self,
        capacity: int,
        *,
        pinned: Array | None = None,
        dist_fn: DistanceFnCallable | DistanceFnLiteral = "euclidean",
        sim_fn: SimilarityFnCallable | SimilarityFnLiteral = "identity",
        loss_fn: LossFnCallable | LossFnLiteral = "power",
        p: float | int = -1,
        k_neighbors: int | float = 5,  # TODO: add heuristic for better default
        sim_min: float = -jnp.inf,
        sim_max: float = +jnp.inf,
        maximize: bool = False,
        dtype: DTypeLike | None = None,
    ):
        """
        Initialize the picker.
        """

        self.capacity: int = capacity

        self.pinned = pinned

        self.dist_fn: DistanceFnCallable = get_function_from_literal(
            dist_fn, module="moll.measures._distance"
        )
        self.sim_fn: SimilarityFnCallable = get_function_from_literal(
            sim_fn, module="moll.measures._similarity"
        )

        self.p: float | int = p
        loss_fn = get_function_from_literal(loss_fn, module="moll.measures._loss")
        loss_fn = partial(loss_fn, p=p) if hasarg(loss_fn, "p") else loss_fn
        self.loss_fn: LossFnCallable = loss_fn

        self.k_neighbors: int = self._init_k_neighbors(k_neighbors, capacity)

        self.maximize: bool = maximize

        self.sim_min: float = sim_min
        self.sim_max: float = sim_max

        # Inferred dtype
        self.dtype: DTypeLike | None
        match (dtype, pinned):
            case (None, Array()):
                self.dtype = pinned.dtype
                logger.info(
                    "Picker dtype={} was inferred from pinned vectors", self.dtype
                )
            case _:
                self.dtype = dtype

        self._data: Array | None = None
        self._labels = np.array([None] * capacity, dtype=object)

        self.n_seen: int = 0
        self.n_accepted: int = 0
        self._n_valid: int = 0

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

    def _init_data(self, vector: Array, label=None):
        """
        Initialize the picker with the first vector.
        """
        source = self.pinned if self.pinned is not None else vector
        dim = source.shape[0]

        if self.dtype is None:
            self.dtype = source.dtype
            logger.info(
                "Picker dtype={} was inferred from the first vector", self.dtype
            )

        self._data = jnp.zeros((self.capacity, dim), dtype=self.dtype)
        self._data = self._data.at[0].set(vector)
        self._labels[0] = label

        self._n_valid += 1
        self.n_accepted += 1
        self.n_seen += 1

    def _convert_labels(self, labels: Indexable[Hashable]) -> NDArray:
        """
        Convert labels to NumPy array.
        """
        if isinstance(labels, np.ndarray):
            return labels

        # If we create the array with a tuple-like label, we get a 2D array.
        # Besides, an empty 1D array is created explicitly, and then it can be filled
        array = np.empty(len(labels), dtype=object)  # init 1D array
        array[:] = labels  # fill it

        return array

    def _update_labels(
        self,
        updated_idxs: Array,
        acceptance_mask: Array,
        labels: NDArray,
    ):
        idxs = updated_idxs[acceptance_mask]

        # If we create the array with a tuple-like label, we get a 2D array.
        # Besides, an empty 1D array is created explicitly, and then it can be filled
        _labels = labels  # copy value
        labels = np.empty(len(labels), dtype=object)  # init 1D array
        labels[:] = _labels  # fill it

        # Update labels
        self._labels[idxs] = labels[acceptance_mask]

    @staticmethod
    def _convert_data(data: Iterable, dtype: DTypeLike | None) -> Array:
        """
        Convert data to JAX array.
        """
        if not isinstance(data, Array | list):
            data = list(data)

        return jnp.array(data, dtype=dtype)

    def partial_fit(
        self, vectors: Iterable, labels: Indexable[Hashable] | None = None
    ) -> int:
        """
        Add a batch of vectors to the picker.
        """
        # Convert to JAX array if needed
        vectors = self._convert_data(vectors, dtype=self.dtype)

        batch_size = len(vectors)
        n_accepted = 0

        # Check labels

        if labels is None:
            labels = np.arange(self.n_seen, self.n_seen + len(vectors))
        elif len(labels) != batch_size:
            raise ValueError(
                f"Expected number of labels={len(labels)} to match batch_size={batch_size}"
            )
        else:
            labels = self._convert_labels(labels)

        # Init internal data storage with a first vector if picker is empty

        if was_empty := self.is_empty():
            self._init_data(vectors[0], labels[0])
            n_accepted += 1

            # Continue with the rest of the vectors
            vectors = vectors[1:]
            labels = labels[1:]

        # Process remaining vectors

        if vectors.shape[0] > 0:
            (
                data_updated,
                updated_idxs,
                acceptance_mask,
                n_appended,
                n_updated,
            ) = update_vectors(
                X=self._data,
                X_pinned=self.pinned,
                xs=vectors,
                dist_fn=self.dist_fn,
                sim_fn=self.sim_fn,
                loss_fn=self.loss_fn,
                k_neighbors=self.k_neighbors,
                sim_min=self.sim_min,
                sim_max=self.sim_max,
                n_valid=self._n_valid,
                maximize=self.maximize,
            )

            # Update vectors data
            self._data = data_updated

            # Update counters
            self._n_valid += int(n_appended)
            n_accepted += int(n_appended) + int(n_updated)

            # Update labels
            self._update_labels(updated_idxs, acceptance_mask, labels)

        n_accepted = min(n_accepted, self.capacity)
        self.n_accepted += n_accepted - was_empty
        self.n_seen += batch_size - was_empty

        return n_accepted

    def fit(self, X, y=None):
        """
        Pick a subset of vectors based on their similarity.
        """
        self.partial_fit(X, y)
        return self

    def add(self, vector: Array, label: Hashable | None = None) -> bool:
        """
        Add a vector to the picker.
        """
        n_accepted = self.partial_fit(
            vectors=[vector],
            labels=[label] if label else None,  # type: ignore
        )
        is_accepted = n_accepted > 0
        return is_accepted

    def transform(self, X):
        return self.vectors

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def warm(self, vectors: Array, labels: Indexable[Hashable] | None = None):
        """
        Initialize the picker with a set of vectors.
        """
        assert self.is_empty()

        batch_size = vectors.shape[0]
        assert batch_size <= self.capacity

        if labels:
            assert len(labels) == batch_size
        else:
            labels = np.arange(len(vectors))

        self._init_data(vectors[0], labels[0])
        self._data = self._data.at[1:batch_size].set(vectors[1:])  # type: ignore
        self._labels[1:batch_size] = labels[1:]

        self.n_accepted += batch_size - 1
        self.n_seen += batch_size - 1
        self._n_valid += batch_size - 1

    @property
    def n_rejected(self) -> int:
        """
        Return the number of rejected vectors.
        """
        return self.n_seen - self.n_accepted

    @property
    def size(self) -> int:
        """
        Return the number of vectors in the picker.
        """
        assert self._n_valid <= self.capacity
        return self._n_valid

    def is_full(self) -> bool:
        """
        Return whether the picker is full or not.
        """
        if self.is_empty():
            return False
        return self.capacity == self.size

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
        return self._labels[: self._n_valid].tolist()

    @property
    def vectors(self) -> Array | None:
        """
        Return the currently picked vectors.
        """
        if self._data is None:
            return None
        return self._data[: self._n_valid]

    @property
    def dim(self) -> int | None:
        """
        Return the dimension of the vectors.
        """
        if (data := self._data) is not None:
            return data.shape[1]
        return None

    @property
    def n_pinned(self) -> int:
        """
        Return the number of pinned vectors.
        """
        return 0 if self.pinned is None else self.pinned.shape[0]
