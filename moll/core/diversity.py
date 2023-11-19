from typing import Callable

import jax.numpy as jnp
from loguru import logger

from moll.core.distance import add_points_to_bag

__all__ = ["OnlineDiversityPicker"]


class OnlineDiversityPicker:
    """
    Greedy algorithm for picking a diverse subset of points in an online fashion.
    """

    def __init__(
        self,
        capacity: int,
        dist_fn: Callable,
        k_neighbors: int = 5,
        p: float = 1.0,
        threshold: float = 0.0,
        dtype: jnp.dtype = None,
    ):
        self.capacity: int = capacity
        self.dist_fn = dist_fn
        self.k_neighbors = k_neighbors
        self.p = p
        self.threshold = threshold

        self._data: jnp.ndarray | None = None
        self.dtype: jnp.dtype | None = dtype
        self._labels: list = [None] * capacity

        self.n_seen: int = 0
        self.n_accepted: int = 0
        self.n_valid_points: int = 0

    def _init(self, point: jnp.ndarray, label=None) -> int:
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

    def extend(self, points: jnp.ndarray, labels=None) -> int:
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
            self._init(points[0], labels[0])
            points = points[1:]
            labels = labels[1:]
            n_accepted += 1

        # Process remaining points

        if points.shape[0] > 0:
            update_idxs, data_updated, acceptance_mask = add_points_to_bag(
                X=self._data,
                xs=points,
                dist_fn=self.dist_fn,
                k_neighbors=self.k_neighbors,
                power=self.p,
                threshold=self.threshold,
                n_valid_points=self.n_valid_points,
            )

            self._data = data_updated

            last_valid_idx = self.n_valid_points - 1
            n_updated = (
                ((update_idxs >= 0) & (update_idxs <= last_valid_idx)).sum().item()
            )
            n_appended = (update_idxs > last_valid_idx).sum().item()
            self.n_valid_points += n_appended
            n_accepted += n_appended + n_updated

            # Update labels
            for updated_idx, label in zip(update_idxs, labels):
                if updated_idx >= 0:
                    self._labels[updated_idx] = label

        n_accepted = min(n_accepted, self.capacity)
        self.n_accepted += n_accepted - was_empty
        self.n_seen += batch_size - was_empty

        return n_accepted

    def append(self, point, label=None) -> bool:
        """
        Add a point to the picker.
        """
        points = jnp.array([point])
        labels = [label] if label else None
        n_accepted = self.extend(points, labels)
        is_accepted = n_accepted > 0
        return is_accepted

    def fast_init(self, points: jnp.ndarray, labels=None) -> int:
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

        self._init(points[0], labels[0])

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
