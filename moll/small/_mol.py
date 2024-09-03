"""
This module contains the `Molecule` class.
"""

from collections import defaultdict
from collections.abc import Generator, Hashable
from dataclasses import dataclass
from functools import cache
from typing import TypeVar

import numpy as np
from numpy.typing import DTypeLike, NDArray
from public import public
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors  # type: ignore

from ..typing import SMILES, FingerprintLiteral, RDKitAtom, RDKitMol
from ..utils import fold

try:
    from typing import Self
except ImportError:
    Self = TypeVar("Self")


@cache
def _fp_generator(kind: FingerprintLiteral, radius: int, size: int):
    """
    Return cached fingerprint generator.
    """
    match kind:
        case "morgan":
            return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=size)
        case _:
            raise ValueError(f"Unknown fingerprint kind: {kind}")


_HETERO_ATOMS = set("CH")


@dataclass
class Atom:
    rdkit_atom: RDKitAtom

    @property
    def symbol(self) -> str:
        return self.rdkit_atom.GetSymbol()

    @property
    def index(self) -> int:
        return self.rdkit_atom.GetIdx()

    @property
    def mass(self) -> float:
        return self.rdkit_atom.GetMass()

    @property
    def n_atomic(self) -> int:
        return self.rdkit_atom.GetAtomicNum()

    @property
    def n_protons(self) -> int:
        return self.n_atomic

    @property
    def n_neutrons(self) -> int:
        return self.n_mass - self.n_protons

    @property
    def n_mass(self) -> int:
        return round(self.mass)

    @property
    def n_charge(self) -> int:
        return self.rdkit_atom.GetFormalCharge()

    def is_isotope(self) -> bool:
        return self.rdkit_atom.GetIsotope() != 0

    def is_aromatic(self) -> bool:
        return self.rdkit_atom.GetIsAromatic()

    def is_chiral(self) -> bool:
        return self.rdkit_atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED

    def is_hetero(self) -> bool:
        return self.symbol not in _HETERO_ATOMS

    def __repr__(self) -> str:
        discriminator = f"{self.n_mass}-" if self.is_isotope() else ""
        return f"<{discriminator}{self.symbol}#{self.index}>"


@public
class Molecule:
    def __init__(self, rdkit_mol=None, label: Hashable = None):
        self._rdkit_mol = rdkit_mol
        self.label = label

    @classmethod
    def from_smiles(cls, smiles: SMILES, label: Hashable = None) -> Self:
        """
        Create molecule from SMILES string.

        >>> mol = Molecule.from_smiles("C=C")
        >>> mol.n_atoms()
        6

        >>> Molecule.from_smiles("Boo!")
        Traceback (most recent call last):
            ...
        ValueError: Could not interpret SMILES: Boo!
        """
        rdkit_mol = Chem.MolFromSmiles(smiles)  # type: ignore
        if rdkit_mol is None:
            raise ValueError(f"Could not interpret SMILES: {smiles}")
        return cls(rdkit_mol=rdkit_mol, label=label)

    @classmethod
    def from_rdkit(cls, rdkit_mol: RDKitMol, label: Hashable = None) -> Self:
        """
        Create molecule from RDKit Mol object.

        Examples:
            >>> from rdkit import Chem
            >>> rdmol = Chem.MolFromSmiles("C=C")
            >>> mol = Molecule.from_rdkit(rdmol)
            >>> mol.n_atoms()
            6

            >>> Molecule.from_rdkit("Boo!")
            Traceback (most recent call last):
                ...
            TypeError: Input is not an RDKit Mol object
        """
        if not isinstance(rdkit_mol, RDKitMol):
            raise TypeError("Input is not an RDKit Mol object")
        return cls(rdkit_mol=rdkit_mol, label=label)

    @property
    def rdkit(self) -> RDKitMol:
        """
        Return the RDKit Mol object.
        """
        return self._rdkit_mol

    def to_fp(
        self,
        kind: FingerprintLiteral,
        *,
        radius: int,
        size: int,
        fold_size: int | None = None,
        dtype: DTypeLike | None = None,
    ) -> NDArray | None:
        """
        Calculate a fingerprint for the molecule.

        Examples:
            >>> mol = Molecule.from_smiles("CCO")

            Generate a Morgan fingerprint:
            >>> mol.to_fp("morgan", radius=2, size=8)
            array([1, 1, 1, 0, 0, 0, 1, 1], dtype=uint8)

            Reduce the size of the fingerprint by folding:
            >>> mol.to_fp("morgan", radius=2, size=1024, fold_size=8)
            array([1, 1, 1, 0, 0, 0, 2, 1], dtype=uint64)

            Convert to a binary vector:
            >>> mol.to_fp("morgan", radius=2, size=1024, fold_size=8, dtype=bool)
            array([ True,  True,  True, False, False, False,  True,  True])
        """
        # FIXME: bad return type
        fp_gen = _fp_generator(kind, radius, size)
        fp = fp_gen.GetFingerprintAsNumPy(self.rdkit)

        fp = np.asarray(fp, dtype)
        if fold_size is not None:
            return fold(fp, dim=fold_size, dtype=dtype)
        return fp

    def n_atoms(self, implicit=True) -> int:  # TODO: add heavy
        """
        Return the number of atoms.

        Examples:
            By default, implicit hydrogens are counted:
            >>> water = Molecule.from_smiles("O")
            >>> water.n_atoms()
            3

            Set `implicit=False` to count only explicit hydrogens:
            >>> Molecule.from_smiles("O").n_atoms(implicit=False)
            1

            Analogously:
            >>> benzene = Molecule.from_smiles("c1ccccc1")
            >>> benzene.n_atoms()
            12

            Benzene with explicit hydrogens:
            >>> Molecule.from_smiles("c1ccccc1").n_atoms(implicit=False)
            6
        """
        return self.rdkit.GetNumAtoms(onlyExplicit=not implicit)

    def n_bonds(self, implicit=True) -> int:  # TODO: change to heavy?
        """
        Return the number of bonds.

        Examples:
            By default, implicit hydrogens are counted:
            >>> water = Molecule.from_smiles("O")
            >>> water.n_bonds()
            2

            >>> benzene = Molecule.from_smiles("c1ccccc1")
            >>> benzene.n_bonds()
            12

            Set `implicit=False` to count only explicit hydrogens:
            >>> water.n_bonds(implicit=False)
            0

            >>> benzene.n_bonds(implicit=False)
            6
        """
        return self.rdkit.GetNumBonds(onlyHeavy=not implicit)

    def n_rings(self) -> int:
        """
        Return the number of rings.

        Examples:
            Water:
            >>> Molecule.from_smiles("O").n_rings()
            0

            Benzene:
            >>> Molecule.from_smiles("c1ccccc1").n_rings()
            1

            Naphthalene:
            >>> Molecule.from_smiles("c1ccc2ccccc2c1").n_rings()
            2
        """
        return self.rdkit.GetRingInfo().NumRings()

    def n_aromatics(self) -> int:
        return self.rdkit.GetNumAromaticAtoms()

    def weight(self) -> float:
        """
        Return the molecular weight.

        Examples:
            Water:
            >>> Molecule.from_smiles("O").weight()
            18.01...

            Benzene:
            >>> Molecule.from_smiles("c1ccccc1").weight()
            78.04...
        """
        return Chem.rdMolDescriptors.CalcExactMolWt(self.rdkit)

    def counts(self, implicit=True) -> dict[str, int]:
        """
        Return the counts of different atom types.

        Examples:
            Water:
            >>> Molecule.from_smiles("O").counts()
            {'O': 1, 'H': 2}

            >>> Molecule.from_smiles("O").counts(implicit=False)
            {'O': 1}
        """
        if implicit:
            mol = Chem.Mol(self.rdkit)  # type: ignore
            mol = Chem.AddHs(mol)
        else:
            mol = self.rdkit
        atom_counts = defaultdict(int)
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            atom_counts[symbol] += 1
        return dict(atom_counts)

    def atoms(self, implicit=True) -> Generator[Atom, None, None]:
        """
        Iterate over atoms.

        Examples:
            Water:
            >>> list(Molecule.from_smiles("O").atoms())
            [<O#0>, <H#1>, <H#2>]

            Radioactive methane:
            >>> list(Molecule.from_smiles("C[2H]").atoms())
            [<C#0>, <2-H#1>, <H#2>, <H#3>, <H#4>]

            Explicit hydrogens:
            >>> list(list(Molecule.from_smiles("C[2H]").atoms(implicit=False)))
            [<C#0>, <2-H#1>]
        """
        if implicit:
            mol = Chem.Mol(self.rdkit)
            mol = Chem.AddHs(mol)
        else:
            mol = self.rdkit

        for atom in mol.GetAtoms():
            yield Atom(atom)

    def n_heteros(self) -> int:
        """
        Return the number of heteroatoms.

        Examples:
            Furan:
            >>> Molecule.from_smiles("O1cccc1").n_heteros()
            1

            Benzene:
            >>> Molecule.from_smiles("c1ccccc1").n_heteros()
            0

            Imidazole:
            >>> Molecule.from_smiles("C1=CN=CN1").n_heteros()
            2
        """
        return sum(atom.is_hetero() for atom in self.atoms())

    def is_aromatic(self) -> bool:
        """
        Return whether the molecule is aromatic.

        Examples:
            Benzene:
            >>> Molecule.from_smiles("c1ccccc1").is_aromatic()
            True

            Cyclohexane:
            >>> Molecule.from_smiles("C1CCCCC1").is_aromatic()
            False
        """
        return any(atom.is_aromatic() for atom in self.atoms())

    def is_chiral(self) -> bool:
        """
        Return whether the molecule is chiral.

        Examples:
            Chiral carbon:
            >>> Molecule.from_smiles("C[C@H](F)Cl").is_chiral()
            True

            Non-chiral carbon:
            >>> Molecule.from_smiles("CC(F)Cl").is_chiral()
            False
        """
        return any(atom.is_chiral() for atom in self.atoms())
