"""
Online algorithm for picking a subset of points based on their distance.
"""

from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array
from loguru import logger

from .online_add import update_points

__all__ = ["OnlineDiversityPicker"]


class OnlineDiversityPicker:
    """
    Greedy algorithm for picking a diverse subset of points in an online fashion.
    """

    def __init__(
        self,
        capacity: int,
        similarity_fn: Callable[[Array, Array], float],
        *,
        potential_fn: Callable[[float], float]
        | Literal["power", "exp", "lj"] = "power",
        p: float = -1.0,
        k_neighbors: int | float = 5,
        threshold: float = 0.0,
        dtype: jnp.dtype | None = None,
    ):
        self.capacity = capacity
        self.similarity_fn = similarity_fn

        self.k_neighbors = self._init_k_neighbors(k_neighbors, capacity)

        self.p = p
        self.potential_fn = self._init_potential_fn(potential_fn, self.p)

        self.threshold = threshold

        self._data: jnp.ndarray | None = None
        self.dtype: jnp.dtype | None = dtype
        self._labels: list = [None] * capacity

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

    def _init_potential_fn(self, potential_fn, p) -> Callable[[float], float]:
        match potential_fn:
            case "power":
                potential_fn = lambda d: d**p
            case "exp":
                potential_fn = lambda d: jnp.exp(p * d)
            case "lj":
                potential_fn = lambda d: (p / d) ** 12 - (p / d) ** 6
                logger.warning(
                    "Consider defining a suitable parameter `p` for the Lennard-Jones potential."
                )
        return potential_fn

    def _init_data(self, point: jnp.ndarray, label=None):
        """
        Initialize the picker with the first point.
        """

        dim = point.shape[0]
        self.dtype = point.dtype
        self._data = jnp.zeros((self.capacity, dim), dtype=self.dtype)
        self._data = self._data.at[0].set(point)
        self._labels[0] = label

        self.n_valid_points += 1
        self.n_accepted += 1
        self.n_seen += 1

    def update(self, points: jnp.ndarray, labels=None) -> int:
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
            update_idxs, data_updated, acceptance_mask = update_points(
                X=self._data,
                xs=points,
                similarity_fn=self.similarity_fn,
                potential_fn=self.potential_fn,
                k_neighbors=self.k_neighbors,
                threshold=self.threshold,
                n_valid_points=self.n_valid_points,
            )

            self._data = data_updated

            last_valid_idx = self.n_valid_points - 1
            n_updated = (
                ((update_idxs >= 0) & (update_idxs <= last_valid_idx)).sum().item()
            )
            n_added = (update_idxs > last_valid_idx).sum().item()
            self.n_valid_points += n_added
            n_accepted += n_added + n_updated

            # Update labels
            for updated_idx, label in zip(update_idxs, labels, strict=True):
                if updated_idx >= 0:
                    self._labels[updated_idx] = label

        n_accepted = min(n_accepted, self.capacity)
        self.n_accepted += n_accepted - was_empty
        self.n_seen += batch_size - was_empty

        return n_accepted

    def add(self, point, label=None) -> bool:
        """
        Add a point to the picker.
        """
        points = jnp.array([point])
        labels = [label] if label else None
        n_accepted = self.update(points, labels)
        is_accepted = n_accepted > 0
        return is_accepted

    def warm(self, points: jnp.ndarray, labels=None):
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

        self._data = self._data.at[1:batch_size].set(points[1:])
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
        return self._labels[: self.n_valid_points]

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
