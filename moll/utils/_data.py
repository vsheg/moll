"""
Utilities for working with data.
"""

import itertools
import os
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Literal, TypeVar

from public import public

from ..typing import OneOrMany

## -------------------------------------------------------------------------- ##

D = TypeVar("D")
R = TypeVar("R")


@public
def map_concurrently(
    fn: Callable[[D], R],
    data: Iterable[D],
    *,
    n_workers: int | None = None,
    proc: bool = False,
    exception_fn: Callable[[Exception], Any]
    | Literal["ignore", "raise"]
    | None = "raise",
) -> Generator[R | None, None, None]:
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
        >>> list(map_concurrently(bad_fn, range(5), exception_fn="ignore"))
        [0.0, 1.0, 3.0, 4.0]

        If `exception_fn=None`, `None` is yielded instead of the result:
        >>> list(map_concurrently(bad_fn, range(5), exception_fn=None))
        [0.0, 1.0, None, 3.0, 4.0]

        Exceptions can be handled in a custom way:
        >>> def const(e):
        ...     return 42
        >>> list(map_concurrently(bad_fn, range(5), exception_fn=const))
        [0.0, 1.0, 42, 3.0, 4.0]

        By default, the number of workers is equal to the number of CPU cores,
        multithreading is used. Set `proc=True` to enable multiprocessing:
        >>> from math import factorial
        >>> list(map_concurrently(factorial, range(10, 15), proc=True))
        [3628800, 39916800, 479001600, 6227020800, 87178291200]
    """

    # Determine number of workers
    if n_workers is None:
        n_workers = os.cpu_count()

    # Choose executor
    executor_class = ProcessPoolExecutor if proc else ThreadPoolExecutor

    # Init executor
    with executor_class(max_workers=n_workers) as executor:
        # Submit tasks to executor
        futures = [executor.submit(fn, args) for args in data]

        # Iterate over futures
        for future in futures:
            try:
                # If everything is ok, yield result
                yield future.result()
            except Exception as e:
                # If something went wrong, handle exception
                match exception_fn:
                    case "raise":
                        raise e
                    case "ignore":
                        continue
                    case None:
                        yield None
                    case _ if callable(exception_fn):
                        yield exception_fn(e)
                    case _:
                        raise ValueError(
                            f"Invalid exception handler: {exception_fn}"
                        ) from e


## -------------------------------------------------------------------------- ##

D = TypeVar("D")
R = TypeVar("R")


@public
def iter_transpose(
    data: Iterable[D], collate_fn: Callable[[Iterable[D]], R] = tuple
) -> Generator[R, None, None]:
    """
    Transpose an iterable of iterables.

    Examples:
        >>> list(iter_transpose([[1, 2, 3], [4, 5, 6]]))
        [(1, 4), (2, 5), (3, 6)]

        >>> list(iter_transpose([[1, 2, 3], [4, 5, 6]], collate_fn=sum))
        [5, 7, 9]
    """
    yield from map(collate_fn, zip(*data, strict=True))


## -------------------------------------------------------------------------- ##

D = TypeVar("D")
B = TypeVar("B", bound=Iterable)
T = TypeVar("T")


@public
def iter_batches(
    data: Iterable[D],
    batch_size: int,
    *,
    collate_fn: Callable[[Iterable[D]], B] = list,
    filter_fn: Callable[[D], bool] | None = None,
    transform_fn: Callable[[B], T] | Literal["transpose"] | None = None,
) -> Generator[OneOrMany[B], None, None]:
    """
    Split an iterable into batches.

    Examples:
        >>> list(iter_batches(range(10), 3))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

        >>> list(iter_batches([], 3))
        []

        >>> list(iter_batches(range(4), 5, collate_fn=tuple))
        [(0, 1, 2, 3)]

        >>> list(iter_batches(range(10), 3, collate_fn=sum))
        [3, 12, 21, 9]

        Filter function can be applied before batching:
        >>> list(iter_batches(range(10), 3, filter_fn=lambda n: n % 2 == 0))
        [[0, 2, 4], [6, 8]]

        If single data item has heterogeneous type, `transform_fn="transpose"` can be used to
        split it into batches of homogeneous type:
        >>> data = [(1, "one"), (2, "two"), (3, "three"), (4, "four"), (5, "five")]
        >>> for num, word in iter_batches(data, 2, transform_fn="transpose"):
        ...     print(" plus ".join(word), "is", sum(num))
        one plus two is 3
        three plus four is 7
        five is 5

        When `transform_fn="transpose"`, *tuples of batches* are yielded rather than a single
        batch and `collate_fn` is applied individually to each batch.
        >>> list(iter_batches(data, 2, transform_fn="transpose", collate_fn=list))
        [([1, 2], ['one', 'two']), ([3, 4], ['three', 'four']), ([5], ['five'])]
    """
    data = iter(data)
    if filter_fn is not None:
        data = filter(filter_fn, data)
    while batch := list(itertools.islice(data, batch_size)):
        match transform_fn:
            case None:
                yield collate_fn(batch)
            case "transpose":
                yield tuple(iter_transpose(batch, collate_fn))
            case _ if callable(transform_fn):
                yield tuple(transform_fn(collate_fn(batch)))


## -------------------------------------------------------------------------- ##


def identity(x):
    return x


def pack_values(values: Any) -> tuple:
    """
    Wrap single values in a tuple.

    >>> pack_values((1, 2, 3))
    (1, 2, 3)

    >>> pack_values(1)
    (1,)
    """

    match values:
        case Iterable():
            return tuple(values)
        case _:
            return (values,)


def unpack_values(values: Any) -> Any:
    """
    Drop tuple if it contains a single element.

    >>> unpack_values((1, 2, 3))
    (1, 2, 3)

    >>> unpack_values(range(3))
    (0, 1, 2)

    >>> unpack_values((1,))
    1

    >>> unpack_values(1)
    1
    """
    match values:
        case [value]:
            return value
        case Iterable():
            return tuple(values)
        case _:
            return values


def compose_fns(*fns: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Compose multiple functions into a single function.

    Examples:
        >>> def add_one(x):
        ...     return x + 1
        >>> compose_fns(add_one, add_one, add_one)(2)
        5

        Functions are applied from right to left:
        >>> def square(x):
        ...     return x**2
        >>> def minus(x):
        ...     return -x
        >>> compose_fns(minus, add_one, square)(2)
        -5

        If no functions are provided, identity function is returned:
        >>> compose_fns()(2)
        2

        Function must have compatible number of arguments:
        >>> def add(x, y):
        ...     return x + y
        >>> compose_fns(minus, add)(2, 3)
        -5

        >>> compose_fns(add, minus)(2)
        Traceback (most recent call last):
            ...
        TypeError: add() missing 1 required positional argument: 'y'

        Default arguments can fix this:
        >>> def add(x, y=0):
        ...     return x + y
        >>> compose_fns(add, minus)(2)
        -2

        Single-element tuples are unpacked by default:
        >>> def f(x):
        ...     return (x,)
        >>> compose_fns(f, f, f)(10)
        10
    """

    if not fns:
        return identity

    def composition(*args):
        for fn in reversed(fns):
            args = pack_values(fn(*args))
        return unpack_values(args)

    return composition
