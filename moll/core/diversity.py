from typing import Iterable

import jax.numpy as jnp

from moll.core.distance import add_point_to_bag, add_points_to_bag, is_in_bag

__all__ = ["OnlineDiversityPicker"]


class OnlineDiversityPicker:
    def __init__(
        self,
        capacity: int,
        dist_fn: callable,
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
        self._labels: list = []

        self.n_seen: int = 0
        self.n_accepted: int = 0
        self.n_valid_points: int = 0

    def _fill_if_not_full(self, point: jnp.ndarray) -> (bool, int):
        """
        Add point if pickier is not filed yet.
        """

        idx = None

        # If it's a first point, just add it
        if self.is_empty():
            self._data = jnp.array([point])
            return True, 0

        # If it's not a first point, check if it's already in the bag
        if is_accepted := not is_in_bag(point, self._data):
            self._data = jnp.vstack([self._data, point])
            idx = len(self._data) - 1

        return is_accepted, idx

    def _create_label(self) -> int:
        """
        Return the next label.
        """
        return self.n_seen

    def _add_label_for_accepted_point(self, label, idx) -> None:
        """
        Add a label to the picker.
        """

        label = label or self._create_label()

        assert label not in self._labels, f"Label `{label}` is already in the picker"

        if (idx is None) or (idx == len(self._labels)):
            self._labels.append(label)
        else:
            self._labels[idx] = label

    def append(self, point, label=None, return_idx=False) -> bool:
        """
        Add a point to the picker.
        """

        # If we haven't seen enough points, just add the point to the data rejecting duplicates
        if not self.is_full():
            is_accepted, old_idx = self._fill_if_not_full(point)
        else:
            # If we have seen enough points, decide whether to add the point or not
            self._data, is_accepted, old_idx = add_point_to_bag(
                x=point,
                X=self._data,
                dist_fn=self.dist_fn,
                k_neighbors=self.k_neighbors,
                threshold=self.threshold,
                power=self.p,
            )

            old_idx = None if old_idx < 0 else old_idx

        # Update the number of viewed points
        self._update_counters(is_accepted, label, old_idx)

        if (old_idx is not None) and old_idx >= self.n_valid_points:
            self.n_valid_points += 1

        # Return whether the point was accepted or not
        if return_idx:
            return is_accepted, old_idx
        return is_accepted

    def extend(self, points: jnp.ndarray, labels=None, *, pairs=None) -> int:
        """
        Add a batch of points to the picker.
        """

        # assert pairs is None or (
        #     points is None and labels is None
        # ), "`pairs` are provided, then `points` and `labels` must be None"

        if not labels:
            labels = [None] * len(points)

        batch_size = len(points)
        n_accepted = 0
        was_init = False

        if self.is_empty():
            # compute dim and init data container, add first point
            dim = points.shape[1]
            self._data = jnp.zeros((self.capacity, dim))
            self._data = self._data.at[0].set(points[0])

            self.n_valid_points += 1
            was_init = True  # to return correct n_accepted

            points = points[1:]
            labels = labels and labels[1:]

        if points.shape[0] > 0:
            changed_item_idxs, data_updated, acceptance_mask = add_points_to_bag(
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
                ((changed_item_idxs >= 0) & (changed_item_idxs <= last_valid_idx))
                .sum()
                .item()
            )
            n_appended = (changed_item_idxs > last_valid_idx).sum().item()
            self.n_valid_points += n_appended

            n_accepted = n_appended + n_updated

        n_accepted += was_init
        n_accepted = min(n_accepted, self.capacity, batch_size)
        self.n_accepted += n_accepted
        self.n_seen += batch_size

        return n_accepted

    def _update_counters(self, is_accepted: bool, label=None, label_idx=None):
        """
        Update the counters.
        """
        if is_accepted:
            self.n_accepted += 1
            self._add_label_for_accepted_point(label, label_idx)
        self.n_seen += 1

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
        # TODO: assert self.n_valid_points <= self.capacity
        return self.capacity == self.size()

    def is_empty(self) -> bool:
        """
        Return whether the picker is empty or not.
        """
        data = self._data
        return (data is None) or (data.shape[0] == 0)

    @property
    def labels(self) -> list:
        """
        Return the currently picked labels.
        """
        return self._labels

    @property
    def points(self) -> list:
        """
        Return the currently picked points.
        """
        return self._data

    @property
    def dim(self) -> int | None:
        """
        Return the dimension of the points.
        """
        if (data := self._data) is not None:
            return data.shape[1]
        return None
