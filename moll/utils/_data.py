"""
Utilities for working with data.
"""

import itertools
import os
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Literal, TypeVar

from public import public

## -------------------------------------------------------------------------- ##

T = TypeVar("T")
D = TypeVar("D")


@public
def map_concurrently(
    fn: Callable[[D], T],
    data: Iterable[D],
    *,
    n_workers: int | None = None,
    proc: bool = False,
    exception_handler: Callable[[Exception], Any]
    | Literal["ignore", "raise"]
    | None = "raise",
) -> Generator[T | None, None, None]:
    """
    Apply a function to each item in an iterable in parallel.

    Examples:
        >>> def square(x):
        ...     return x**2
        >>> list(map_concurrently(square, range(10)))
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

        By default, exceptions are raised:
        >>> bad_fn = lambda x: (x - 2) / (x - 2) * x
        >>> accumulator = []
        >>> for result in map_concurrently(bad_fn, range(5)):
        ...     accumulator.append(result)
        Traceback (most recent call last):
            ...
        ZeroDivisionError: division by zero

        All computations before the exception are always returned:
        >>> accumulator
        [0.0, 1.0]

        Exceptions can be easily ignored:
        >>> list(map_concurrently(bad_fn, range(5), exception_handler="ignore"))
        [0.0, 1.0, 3.0, 4.0]

        If `exception_handler=None`, `None` is yielded instead of the result:
        >>> list(map_concurrently(bad_fn, range(5), exception_handler=None))
        [0.0, 1.0, None, 3.0, 4.0]

        Exceptions can be handled in a custom way:
        >>> def const(e):
        ...     return 42
        >>> list(map_concurrently(bad_fn, range(5), exception_handler=const))
        [0.0, 1.0, 42, 3.0, 4.0]
    """

    # Determine data length

    # Determine number of workers
    if n_workers is None:
        n_workers = os.cpu_count()

    # Choose executor
    executor_class = ProcessPoolExecutor if proc else ThreadPoolExecutor

    # Init executor
    with executor_class(max_workers=n_workers) as executor:
        # Submit tasks to executor
        futures = [executor.submit(fn, item) for item in data]

        # Iterate over futures
        for future in futures:
            try:
                # If everything is ok, yield result
                yield future.result()
            except Exception as e:
                # If something went wrong, handle exception
                match exception_handler:
                    case "raise":
                        raise e
                    case "ignore":
                        continue
                    case None:
                        yield None
                    case _ if callable(exception_handler):
                        yield exception_handler(e)
                    case _:
                        raise ValueError(
                            f"Invalid exception handler: {exception_handler}"
                        ) from e


## -------------------------------------------------------------------------- ##

T = TypeVar("T")
V = TypeVar("V")


@public
def iter_batches(
    data: Iterable[T],
    batch_size: int,
    *,
    collation_fn: Callable[[Iterable[T]], V] = list,
) -> Generator[V, None, None]:
    """
    Split an iterable into batches.

    Examples:
        >>> list(iter_batches(range(10), 3))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

        >>> list(iter_batches([], 3))
        []

        >>> list(iter_batches(range(4), 5, collation_fn=tuple))
        [(0, 1, 2, 3)]

        >>> list(iter_batches(range(10), 3, collation_fn=sum))
        [3, 12, 21, 9]
    """
    iterator = iter(data)
    while batch := collation_fn(itertools.islice(iterator, batch_size)):
        yield batch
