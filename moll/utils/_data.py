"""
Utilities for working with data.
"""

import itertools
import os
from collections.abc import Callable, Generator, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from glob import glob
from pathlib import Path
from queue import Queue
from threading import Thread
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
    buffer_size=150_000,
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

    # Init queue
    queue = Queue(maxsize=buffer_size)

    def submit_futures(executor, iterable, n: int):
        """Submit futures to the executor and add them to the queue."""
        for _ in range(n):
            try:
                args = next(iterable)
                future = executor.submit(fn, args)
                queue.put(future)
            except StopIteration:
                break

    # Choose executor
    executor_class = ProcessPoolExecutor if proc else ThreadPoolExecutor

    # Init executor
    with executor_class(max_workers=n_workers) as executor:
        data_iter: Iterator[D] = iter(data)
        submit_futures(executor, data_iter, n=buffer_size)

        while not queue.empty():
            future = queue.get()
            try:
                yield future.result()
            except Exception as e:
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
            submit_futures(executor, data_iter, n=1)


## -------------------------------------------------------------------------- ##

D = TypeVar("D")
R = TypeVar("R")


@public
def iter_transpose(
    data: Iterable[D],
    collate_fn: Callable[[Iterable[D]], R] = tuple,
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


@public
def iter_precompute(
    iterable: Iterable[D],
    n_precomputed: int,
) -> Generator[D, None, None]:
    """
    Function to precompute a number of elements from an iterator.

    Examples:
        >>> numbers = iter_precompute(range(5), n_precomputed=2)
        >>> list(numbers)
        [0, 1, 2, 3, 4]

        >>> numbers = iter_precompute(range(5), n_precomputed=100)
        >>> list(numbers)
        [0, 1, 2, 3, 4]

        >>> numbers = iter_precompute(range(5), n_precomputed=1)
        >>> list(numbers)
        [0, 1, 2, 3, 4]
    """
    if not isinstance(n_precomputed, int) or n_precomputed <= 0:
        raise ValueError("n_precomputed must be a positive integer")

    queue = Queue(maxsize=n_precomputed)
    sentinel = object()

    def precompute():
        for item in iterable:
            queue.put(item)
        queue.put(sentinel)

    precompute_process = Thread(target=precompute)
    precompute_process.start()

    while True:
        next_item = queue.get()
        if next_item is sentinel:
            break
        yield next_item

    precompute_process.join()


## -------------------------------------------------------------------------- ##

D = TypeVar("D")  # Data type
B = TypeVar("B", bound=Iterable)  # Batch type
T = TypeVar("T", bound=Iterable)  # Transformed batch type


@public
def iter_slices(
    data: Iterable[D],
    slice_size: int,
    *,
    collate_fn: Callable[[Iterable[D]], B] = list,
    filter_fn: Callable[[D], bool] | None = None,
    transform_fn: Callable[[B], T] | Literal["transpose"] | None = None,
) -> Generator[OneOrMany[B], None, None]:
    """
    Split an iterable into batches of a given size.

    Examples:
        >>> list(iter_slices(range(10), 3))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

        >>> list(iter_slices([], 3))
        []

        >>> list(iter_slices(range(4), 5, collate_fn=tuple))
        [(0, 1, 2, 3)]

        >>> list(iter_slices(range(10), 3, collate_fn=sum))
        [3, 12, 21, 9]

        Filter function can be applied before batching:
        >>> list(iter_slices(range(10), 3, filter_fn=lambda n: n % 2 == 0))
        [[0, 2, 4], [6, 8]]

        If single data item has heterogeneous type, `transform_fn="transpose"` can be used
        to split it into batches of homogeneous type:
        >>> data = [(1, "one"), (2, "two"), (3, "three"), (4, "four"), (5, "five")]
        >>> for num, word in iter_slices(data, 2, transform_fn="transpose"):
        ...     print(" plus ".join(word), "is", sum(num))
        one plus two is 3
        three plus four is 7
        five is 5

        When `transform_fn="transpose"`, *tuples of batches* are yielded rather than a
        single batch and `collate_fn` is applied individually to each batch.
        >>> list(iter_slices(data, 2, transform_fn="transpose", collate_fn=list))
        [([1, 2], ['one', 'two']), ([3, 4], ['three', 'four']), ([5], ['five'])]

        Custom transformation functions can be used:
        >>> def compute_statistics(batch):
        ...     return [min(batch), max(batch), sum(batch)]
        >>> list(iter_slices(range(10), 3, transform_fn=compute_statistics))
        [(0, 2, 3), (3, 5, 12), (6, 8, 21), (9, 9, 9)]
    """
    data = iter(data)
    if filter_fn is not None:
        data = filter(filter_fn, data)
    while batch := list(itertools.islice(data, slice_size)):
        match transform_fn:
            case None:
                yield collate_fn(batch)
            case "transpose":
                yield tuple(iter_transpose(batch, collate_fn))
            case _ if callable(transform_fn):
                yield tuple(transform_fn(collate_fn(batch)))


## -------------------------------------------------------------------------- ##

T = TypeVar("T")


@public
def iter_lines(
    files: OneOrMany[str],
    skip_rows: int = 0,
    source_fn: Callable[[str], str] | Literal["filename", "stem"] | None = None,
    line_fn: Callable[[str], T] | None | Literal["split"] = None,
) -> Generator[tuple[str, int, str | T | tuple[str]], None, None]:
    """
    Iterate over lines in files.
    """
    files = pack_values(files)

    # Expand globs
    files = itertools.chain.from_iterable(map(glob, files))

    # Handle source label
    match source_fn:
        case None:
            pass
        case "name":
            source_fn = lambda file: Path(file).name
        case "stem":
            source_fn = lambda file: Path(file).stem
        case _ if callable(source_fn):
            source_fn = source_fn
        case _:
            raise ValueError(f"Invalid source label function: {source_fn}")

    # Handle line function
    match line_fn:
        case None:
            pass
        case _ if callable(line_fn):
            line_fn = line_fn
        case "split":
            line_fn = lambda line: tuple(line.split())
        case _:
            raise ValueError(f"Invalid line function: {line_fn}")

    # Iterate over files
    for file in files:
        with Path(file).open() as f:
            # Iterate over lines
            for line_no, line in enumerate(iterable=f):
                if line_no < skip_rows:
                    continue

                line = line.rstrip("\n")

                yield (
                    source_fn(file) if source_fn is not None else file,
                    line_no,
                    line_fn(line) if line_fn is not None else line,
                )


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

    >>> pack_values("hello")
    ('hello',)
    """

    match values:
        case str():
            return (values,)
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
