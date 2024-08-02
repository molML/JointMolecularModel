""" Perform substructure similarity on maximum common substructures (MCS) between two molecules. We define similarity as
the fraction of atoms of the full molecule belonging to the largest common moiety.

Similarity = MCS / N

This quantifes how well parts of the molecule are represented in another (set of) mol(s)

Important:
    - standard implementation is not symmetric, i.e., f(a, b) != f(b, a)
    - Uses the first MCS found by the FMCS algorithm (i.e. repeated MCS's in a molecule don't affect the similarity)


Written by Luke Rossen, adapted by Derek van Tilborg
Eindhoven University of Technology
Aug 2024
"""

from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.rdFMCS import BondCompare, RingCompare


class FMCS():
    """ FMCS wrapper with SMARTS querying """
    def __init__(self):
        self.params = rdFMCS.MCSParameters()
        self.params.BondTyper = BondCompare.CompareOrderExact
        self.params.RingTyper = RingCompare.PermissiveRingFusion
        # actually True, handled in custom call instead.
        self.params.AtomCompareParameters.RingMatchesRingOnly = True
        self.params.AtomCompareParameters.CompleteRingsOnly = True
        self.params.AtomCompareParameters.MatchValences = True
        self.params.AtomCompareParameters.MatchChiralTag = False
        self.params.BondCompareParameters.MatchFusedRings = True
        self.params.BondCompareParameters.MatchFusedRingsStrict = False
        self.params.Timeout = 1
        self.params.MaximizeBonds = True

    def _get_smartsquery(self, mol1: Chem.rdchem.Mol, mol2: Chem.rdchem.Mol) -> str:
        return rdFMCS.FindMCS([mol1, mol2], self.params).smartsString

    def _get_querymol(self, mol1: Chem.rdchem.Mol, mol2: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
        return Chem.MolFromSmarts(self._get_smartsquery(mol1, mol2))

    def __call__(self, mol1: Chem.rdchem.Mol, mol2: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
        return self._get_querymol(mol1, mol2)


class MCSSimilarity:
    """ Compute substructure similarity on maximum common substructures (MCS) between two molecules """
    def __init__(self, **kwargs):
        self.FMCS = FMCS(**kwargs)

    def calc_similarity(self, mol1: Chem.rdchem.Mol, mol2: Chem.rdchem.Mol, symmetric: bool = False) -> float:
        """ MCS similarity

        :param mol1: molecule 1, when symmetric = False, this is the parent molecule
        :param mol2: molecule 2
        :param symmetric: toggles symmetric substructure similarity. Similarly to Jensen-Shannon divergence, we take
        the mean of the two distances such that it is symmetrical
        :return:
        """
        substructure = self.FMCS(mol1, mol2)

        if symmetric:
            return (substructure_similarity(mol1, substructure) + substructure_similarity(mol2, substructure)) / 2
            # return 2 * substructure.GetNumAtoms() / (mol1.GetNumAtoms() + mol2.GetNumAtoms())  # alternative way

        return substructure_similarity(mol1, substructure)

    def __call__(self, *args, **kwargs) -> float:
        self.calc_similarity(*args, **kwargs)


def substructure_similarity(mol: Chem.rdchem.Mol, substructure: Chem.rdchem.Mol) -> float:
    """ Get the Maximal Common Substructure Fraction

    Important note here: if substructure exists more than once in mol, we report fraction only for the first match found.

    :param mol: RDKit mol object of the full molecule
    :param substructure: RDKit mol object of the substructure
    :return: substructure similarity
    """
    return substructure.GetNumAtoms() / mol.GetNumAtoms()
