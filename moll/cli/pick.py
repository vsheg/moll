"""
CLI subcommand for picking algorithms.
"""

import itertools
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, List

import typer

from moll.pick import OnlineVectorPicker
from moll.small import Molecule

from ..typing import DistanceFnLiteral, LossFnLiteral, SimilarityFnLiteral
from ..utils import iter_transpose

# Subcommand for picking algorithms
cli_pick = typer.Typer()

MolFile = Annotated[
    Path,
    typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
]


def _vectorize(mol, vec_spec: str):
    """
    Temporary solution. TODO.
    """
    kind, _, specs = vec_spec.partition("-")

    match kind:
        case "ecfp":
            pattern = r"r(\d+)-d(\d+)"
            match = re.match(pattern, specs)
            if not match:
                raise ValueError(f"Invalid vector spec: {vec_spec}")
            radius, size = map(int, match.groups())
            return mol.to_fp(kind=kind, radius=radius, size=size), mol.label
        case _:
            raise ValueError(f"Unknown vector kind: {kind}")


@cli_pick.command()
def online(
    paths: MolFile,
    capacity: int,
    k_neighbors: int = 5,
    vec_fn: str = "ecfp-r2-d1024",
    dist_fn: str = "one_minus_tanimoto",
    sim_fn: str = "identity",
    loss_fn: str = "power",
    p: float = -1,
):
    """
    Pick molecules using the online algorithm.
    """

    # Now only .smi files are supported
    mols = Molecule.from_smi_file(paths)
    vecs, labels = iter_transpose(_vectorize(mol, vec_fn) for mol in mols)

    picker = OnlineVectorPicker(
        capacity=capacity,
        dist_fn=dist_fn,
        sim_fn=sim_fn,
        loss_fn=loss_fn,
        p=p,
        k_neighbors=k_neighbors,
    )

    picker.fit(vecs, labels)
