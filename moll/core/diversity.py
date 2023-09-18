from typing import Iterable

import jax.numpy as jnp

from moll.core.distance import add_one_point_to_the_bag, add_points_to_bag, is_in_bag

__all__ = ["OnlineDiversityPicker"]


class OnlineDiversityPicker:
    def __init__(
        self,
        n_points: int,
        dist: callable,
        n_neighbors: int = 3,
        threshold: float = 0.0,
    ):
        self.n_points: int = n_points
        self.dist = dist
        self.n_neighbors = n_neighbors
        self.threshold = threshold

        self.data_ = None

        self.n_viewed: int = 0
        self.n_accepted: int = 0

    def _fill_if_not_full(self, point: jnp.ndarray) -> bool:
        """
        Add point if pickier is not filed yet.
        """

        # If it's a first point, just add it
        if self.is_empty():
            self.data_ = jnp.array([point])
            self.n_accepted = 1
            return True

        # If it's not a first point, check if it's already in the bag
        if is_accepted := not is_in_bag(point, self.data_):
            self.data_ = jnp.vstack([self.data_, point])
            self.n_accepted += is_accepted

        return is_accepted

    def append(self, point) -> bool:
        """
        Add a point to the picker.
        """

        # Update the number of viewed points
        self.n_viewed += 1

        # If we haven't seen enough points, just add the point to the data rejecting duplicates
        if not self.is_full():
            is_accepted = self._fill_if_not_full(point)
        else:
            # If we have seen enough points, decide whether to add the point or not
            is_accepted, self.data_ = add_one_point_to_the_bag(
                x=point,
                X=self.data_,
                dist=self.dist,
                n_neighbors=self.n_neighbors,
                threshold=self.threshold,
            )
            self.n_accepted += is_accepted

        # Return whether the point was accepted or not
        return is_accepted

    def fast_init(self, points: jnp.ndarray) -> None:
        """
        Initialize the picker with a batch of points.
        """

        assert len(points) == self.n_points

        self.data_ = jnp.array(points)
        self.n_accepted = len(points)
        self.n_viewed = len(points)

    def extend(self, points: jnp.ndarray | list) -> int:
        """
        Add a batch of points to the picker.
        """

        n_accepted = 0

        # If we haven't seen enough points, just add the points to the data rejecting duplicates
        for idx, point in enumerate(points):
            if self.is_full():
                break
            else:
                is_accepted = self._fill_if_not_full(point)
                n_accepted += is_accepted
                self.n_viewed += 1
        else:
            # If we iterated over all points, return the number of added points
            self.n_accepted += n_accepted
            return n_accepted

        # If there still some point:
        points = points[idx:]

        acceptances, self.data_ = add_points_to_bag(
            xs=points,
            X=self.data_,
            dist=self.dist,
            n_neighbors=self.n_neighbors,
            threshold=self.threshold,
        )

        # Update the number of accepted points
        sum = acceptances.sum().item()
        self.n_accepted += sum
        n_accepted += sum
        self.n_viewed += len(points)

        return n_accepted

    @property
    def n_rejected(self) -> int:
        """
        Return the number of rejected points.
        """
        return self.n_viewed - self.n_accepted

    def is_full(self) -> bool:
        """
        Return whether the picker is full or not.
        """
        if self.is_empty():
            return False
        return self.data_.shape[0] >= self.n_points

    def is_empty(self) -> bool:
        """
        Return whether the picker is empty or not.
        """
        return self.data_ is None

    def online(self, stream: Iterable, batch_size: int = 100) -> None:
        """
        Add points from a stream to the picker.
        """
        pass

    def labels(self) -> list:
        """
        Return the currently picked labels.
        """
        pass

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
