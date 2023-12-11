import warnings
from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any

from public import public

from moll.typing import OneOrMany


@public
def args_support(deco: Callable):
    """
    Decorator to allow a decorator to be used with or without arguments.

    >>> @args_support
    ... def constant_decorator(fn, const=10):
    ...     return lambda: 10
    ...

    >>> @constant_decorator
    ... def fn20():
    ...     return 20
    >>> fn20()
    10

    >>> @constant_decorator(const=30)
    ... def fn40():
    ...     return 40
    >>> fn40()
    10
    """

    @wraps(deco)
    def wrapper(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # Called as @deco
            return deco(args[0])
        else:
            # Called as @deco(...)
            return lambda fn_: deco(fn_, *args, **kwargs)

    return wrapper


@public
def listify(
    fn: Callable[..., Iterable],
) -> Callable:
    """
    Decorator to convert a generator function into a list-returning function.

    >>> @listify
    ... def numbers():
    ...     yield from range(5)
    >>> numbers()
    [0, 1, 2, 3, 4]

    >>> @listify
    ... def empty():
    ...     if False: yield
    >>> empty()
    []
    """

    @wraps(fn)
    def wrapper(*args, **kwargs) -> list:
        return list(fn(*args, **kwargs))

    return wrapper


@public
@args_support
def no_warnings(
    fn: Callable,
    suppress_rdkit=True,
) -> Callable:
    """
    Decorator to suppress warnings in a function.

    >>> import warnings
    >>> @no_warnings
    ... def warn():
    ...     warnings.warn("Boooo!!!", UserWarning)
    >>> warn()

    >>> from rdkit import Chem
    >>> @no_warnings
    ... def warn_rdkit():
    ...     Chem.MolFromSmiles('C1=CC=CC=C1O')
    >>> warn_rdkit()
    """
    RDLogger = None
    if suppress_rdkit:
        from rdkit import RDLogger

    @wraps(fn)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if suppress_rdkit and RDLogger is not None:
                RDLogger.DisableLog("rdApp.*")
            result = fn(*args, **kwargs)
            if suppress_rdkit and RDLogger is not None:
                RDLogger.EnableLog("rdApp.*")
        return result

    return wrapper


@public
@args_support
def no_exceptions(
    fn: Callable,
    exceptions: OneOrMany[type[BaseException]] = Exception,
    default: Any = None,
) -> Callable:
    """
    Decorator to catch exceptions and return a default value instead.

    >>> @no_exceptions(default='Error occurred')
    ... def bad_fn(x):
    ...     return x / 0
    >>> bad_fn(10)
    'Error occurred'

    >>> @no_exceptions(exceptions=ZeroDivisionError)
    ... def bad_fn(x):
    ...     return x / 0
    >>> bad_fn(10)

    >>> @no_exceptions(exceptions=TypeError)
    ... def bad_fn(x):
    ...     return x / 0
    >>> bad_fn(10)
    Traceback (most recent call last):
        ...
    ZeroDivisionError: division by zero
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except exceptions:
            return default

    return wrapper


@public
@args_support
def filter_(
    fn: Callable[..., Iterable],
    cond: Callable[[Any], bool] | None = None,
) -> Callable:
    """
    Decorator to filter iterable items returned by a function by a value.

    >>> @filter_
    ... def numbers():
    ...     return [5, 15, None, 25]
    >>> numbers()
    [5, 15, 25]

    >>> @filter_
    ... def numbers():
    ...     yield from [5, 15, None, 25]
    >>> numbers()
    [5, 15, 25]

    >>> @filter_(cond=lambda x: x > 10)
    ... def numbers():
    ...     return [5, 15, 20, 25]
    >>> numbers()
    [15, 20, 25]
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        return list(filter(cond, fn(*args, **kwargs)))

    return wrapper
