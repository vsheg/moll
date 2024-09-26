from collections.abc import Iterable
from pathlib import Path
from re import match
from types import EllipsisType

from public import public


@public
def check_file(
    path: Path | str,
    exists=True,
    extensions: str | Iterable[str] | None | EllipsisType = ...,
) -> Path:
    """
    Check if the file exists.

    Examples:
        >>> from moll.data import amino_acids_smi

        Check if the file exists:
        >>> check_file(amino_acids_smi)
        PosixPath('moll/data/amino_acids.smi')

        If the file does not exist, raise FileNotFoundError:
        >>> check_file(amino_acids_smi.with_suffix(".txt"))
        Traceback (most recent call last):
        ...
        FileNotFoundError: File moll/data/amino_acids.txt not found.

        Check if the file has a specific extension:
        >>> check_file(amino_acids_smi, extensions=".smi")
        PosixPath('moll/data/amino_acids.smi')

        `ValueError` is raised if the file does not have one of the specified extensions:
        >>> check_file(amino_acids_smi, extensions=[".mol2", ".sdf"])
        Traceback (most recent call last):
        ...
        ValueError: File moll/data/amino_acids.smi should have one of the extensions ['.mol2', '.sdf'].

    """

    if not (path := Path(path)).exists() and exists:
        raise FileNotFoundError(f"File {path} not found.")

    if isinstance(extensions, str):
        extensions = (extensions,)

    match extensions:
        case EllipsisType():
            pass
        case None:
            # check no extensions
            if path.suffix:
                raise ValueError(f"File {path} should not have an extension.")
        case list():
            # match regex
            for ext in extensions:
                if match(ext, path.suffix):
                    break
            else:
                raise ValueError(
                    f"File {path} should have one of the extensions {extensions}."
                )

    return path
