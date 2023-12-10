import warnings
from collections.abc import Callable, Generator
from functools import wraps
from typing import Any

from public import public


@public
def listify(fn: Callable[..., Generator[Any, None, None]]) -> Callable:
    """
    Decorator to convert a generator function into a list-returning function.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs) -> list[Any]:
        return list(fn(*args, **kwargs))

    return wrapper


@public
def no_warnings(fn):
    """
    Decorator to suppress warnings in a function.
    """

    @wraps(fn)
    def new(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)

    return new


@public
def no_exceptions(exceptions=Exception, default=None):
    """
    Decorator to catch exceptions and return a default value instead.
    """

    def deco(func):
        def new(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions:
                return default

        return new

    return deco


@public
def filter_value(fn, value=None):
    """
    Decorator to filter the output of a function by a provided value.
    """

    def wrapper(*args, **kwargs):
        return list(filter(value, fn(*args, **kwargs)))

    return wrapper
