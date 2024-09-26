"""
Data provided for testing, examples, documentation, etc.
"""

from pathlib import Path

_parent = Path(__file__).parent  # is moll/data
_cwd = Path.cwd()  # user's working directory

amino_acids_smi: Path = (_parent / "amino_acids.smi").relative_to(_cwd, walk_up=True)

d_aldopentoses_sdf: Path = (_parent / "d_aldopentoses.sdf").relative_to(
    _cwd, walk_up=True
)

saturated_fatty_acids_mol2 = (_parent / "saturated_fatty_acids.mol2").relative_to(
    _cwd, walk_up=True
)
