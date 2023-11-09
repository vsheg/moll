from typing import Iterable

import jax.numpy as jnp

from moll.core.distance import (
    add_point_to_bag,
    add_points_to_bag,
    is_in_bag,
    keep_last_changes,
)

__all__ = ["OnlineDiversityPicker"]


class OnlineDiversityPicker:
    def __init__(
        self,
        n_points: int,
        dist: callable,
        n_neighbors: int = 5,
        p: float = 1.0,
        threshold: float = 0.0,
    ):
        self.n_points: int = n_points
        self.dist = dist
        self.n_neighbors = n_neighbors
        self.p = p
        self.threshold = threshold

        self.data_ = None
        self.labels = []

        self.n_seen: int = 0
        self.n_accepted: int = 0

    def _fill_if_not_full(self, point: jnp.ndarray) -> (bool, int):
        """
        Add point if pickier is not filed yet.
        """

        idx = None

        # If it's a first point, just add it
        if self.is_empty():
            self.data_ = jnp.array([point])
            return True, 0

        # If it's not a first point, check if it's already in the bag
        if is_accepted := not is_in_bag(point, self.data_):
            self.data_ = jnp.vstack([self.data_, point])
            idx = len(self.data_) - 1

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

        assert label not in self.labels, f"Label `{label}` is already in the picker"

        if (idx is None) or (idx == len(self.labels)):
            self.labels.append(label)
        else:
            self.labels[idx] = label

    def append(self, point, label=None, return_idx=False) -> bool:
        """
        Add a point to the picker.
        """

        # If we haven't seen enough points, just add the point to the data rejecting duplicates
        if not self.is_full():
            is_accepted, old_idx = self._fill_if_not_full(point)
        else:
            # If we have seen enough points, decide whether to add the point or not
            self.data_, is_accepted, old_idx = add_point_to_bag(
                x=point,
                X=self.data_,
                dist=self.dist,
                n_neighbors=self.n_neighbors,
                threshold=self.threshold,
                power=self.p,
            )

            old_idx = None if old_idx < 0 else old_idx

        # Update the number of viewed points
        self._update_counters(is_accepted, label, old_idx)

        # Return whether the point was accepted or not
        if return_idx:
            return is_accepted, old_idx
        return is_accepted

    def fast_init(self, points: jnp.ndarray) -> None:
        """
        Initialize the picker with a batch of points.
        """

        assert len(points) == self.n_points

        self.data_ = jnp.array(points)
        self.n_accepted = len(points)
        self.n_seen = len(points)

    def extend(self, points: jnp.ndarray | list, labels=None, *, pairs=None) -> int:
        """
        Add a batch of points to the picker.
        """

        assert pairs is None or (
            points is None and labels is None
        ), "`pairs` are provided, then `points` and `labels` must be None"

        if pairs:
            points, labels = zip(*pairs)

        changed_item_idxs = -jnp.ones(len(points), dtype=int)

        # STAGE 1: If we haven't seen enough points, just add the points to the data rejecting duplicates

        for idx, point in enumerate(points):
            if self.is_full():
                break
            else:
                label = labels and labels[idx]
                is_accepted, old_idx = self.append(point, label=label, return_idx=True)
                if is_accepted:
                    changed_item_idxs = changed_item_idxs.at[idx].set(old_idx)
        else:
            # If we iterated over all points, return the number of added points
            return (changed_item_idxs >= 0).sum().item()

        # STAGE 2: If there still some point:
        points = points[idx:]

        self.data_, accepted_points_mask2, changed_item_idxs2 = add_points_to_bag(
            xs=points,
            X=self.data_,
            dist=self.dist,
            n_neighbors=self.n_neighbors,
            power=self.p,
            threshold=self.threshold,
        )

        # Update the number of accepted points
        for new_point_idx, (is_accepted, replaced_point_idx) in enumerate(
            zip(accepted_points_mask2, changed_item_idxs2), start=idx
        ):
            label = labels and labels[new_point_idx]
            self._update_counters(is_accepted, label, replaced_point_idx)

        # STAGE 3: Combine the results of the two stages

        changed_item_idxs = changed_item_idxs.at[idx:].set(changed_item_idxs2)

        # If during the first stage some points were added
        if (changed_item_idxs[:idx] >= 0).any():
            # Some points might have been accepted when the picker was not full and then
            # replaced by another point. In that case we counted them twice, we need to
            # decrement the number of accepted points

            # Now `changed_item_idxs` contains changes from both stages
            n_accepted_on_both_stages = (changed_item_idxs >= 0).sum()

            # Keep only the last unique changes
            changed_item_idxs = keep_last_changes(changed_item_idxs)
            n_unique_changes = (changed_item_idxs >= 0).sum()

            # We can't accept more points than the number of points we need to pick
            n_could_be_accepted = min(self.n_points, n_unique_changes)

            # Calculate the number of extra accepted points
            n_extra_accepted = n_accepted_on_both_stages - n_could_be_accepted

            # Decrease the number of accepted points
            self.n_accepted -= n_extra_accepted.item()

        accepted_points_mask = changed_item_idxs >= 0
        n_accepted = accepted_points_mask.sum().item()

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
        return self.data_.shape[0]

    def is_full(self) -> bool:
        """
        Return whether the picker is full or not.
        """
        if self.is_empty():
            return False
        return self.size() >= self.n_points

    def is_empty(self) -> bool:
        """
        Return whether the picker is empty or not.
        """
        return self.data_ is None

    def labels(self) -> list:
        """
        Return the currently picked labels.
        """
        return self.labels

    @property
    def points(self) -> list:
        """
        Return the currently picked points.
        """
        return self.data_

    @property
    def dim(self) -> int:
        """
        Return the dimension of the points.
        """
        if self.is_empty():
            return None
        return self.data_.shape[1]
