from typing import Callable

import jax.numpy as jnp

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
    ):
        self.capacity: int = capacity
        self.dist_fn = dist_fn
        self.k_neighbors = k_neighbors
        self.p = p
        self.threshold = threshold

        self._data: jnp.ndarray | None = None
        self._labels: list = [None] * capacity

        self.n_seen: int = 0
        self.n_accepted: int = 0
        self.n_valid_points: int = 0

    def append(self, point, label=None) -> bool:
        """
        Add a point to the picker.
        """
        points = jnp.array([point])
        labels = [label] if label else None
        n_accepted = self.extend(points, labels)
        is_accepted = n_accepted > 0
        return is_accepted

    def extend(self, points: jnp.ndarray, labels=None) -> int:
        """
        Add a batch of points to the picker.
        """

        if not labels:
            labels = range(self.n_seen, self.n_seen + len(points))

        batch_size = len(points)
        n_accepted = 0

        if self.is_empty():
            # compute dim and init data container, add first point
            dim = points.shape[1]
            dtype = points.dtype
            self._data = jnp.zeros((self.capacity, dim), dtype=dtype)
            self._data = self._data.at[0].set(points[0])
            self._labels[0] = labels[0]

            self.n_valid_points += 1
            n_accepted += 1

            points = points[1:]
            labels = labels[1:]

        if points.shape[0] > 0:
            update_idxs, data_updated, _acceptance_mask = add_points_to_bag(
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
        self.n_accepted += n_accepted
        self.n_seen += batch_size

        return n_accepted

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
