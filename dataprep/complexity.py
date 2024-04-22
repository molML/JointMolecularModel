
import numpy as np
from rdkit import Chem
import torch


def mean_atom_degree(mol):
    return sum([atom.GetDegree() for atom in mol.GetAtoms()])/mol.GetNumAtoms()


def count_rings(mol):
    ri = mol.GetRingInfo()
    return ri.NumRings()


def num_diff_elements(mol):
    return len(set([atom for atom in mol.GetAtoms() if atom.GetSymbol()]))


def mol_complexity(smile: str) -> float:
    """ Calculates molecular complexity based on a made up formula according to me eyeballing some molecules.

    :param smile: SMILES string, e.g.; 'CC(C)C1CN=C(N)C1'
    :return: estimated complexity
    """
    mol = Chem.MolFromSmiles(smile)
    nrings = count_rings(mol) + 1
    diff_heavy_atoms = num_diff_elements(mol)
    moldegree = mean_atom_degree(mol)

    return moldegree * (nrings) * diff_heavy_atoms


def smile_complexity(smile: str) -> float:
    """ Calculates Bertz CT as a measure of molecular complexity

    RDKit: Bertz CT consists of a sum of two terms, one representing the complexity of the bonding, the other
    representing the complexity of the distribution of heteroatoms.

    From S. H. Bertz, J. Am. Chem. Soc., vol 103, 3599-3601 (1981)

    :param smile: SMILES string, e.g.; 'CC(C)C1CN=C(N)C1'
    :return: estimated complexity

    """
    mol = Chem.MolFromSmiles(smile)
    return Chem.GraphDescriptors.BertzCT(mol)


def split_smiles_by_complexity(smiles: list[str], precomputed: str = None, levels: int = 3):

    if precomputed is not None:
        complexity_dict = torch.load(precomputed)
        complexity = [complexity_dict[smi] for smi in smiles]
    else:
        complexity = [smile_complexity(smi) for smi in smiles]

    order_of_complexity = np.argsort(complexity)

    splits = np.array_split(order_of_complexity, levels)
    splits = [np.concatenate(splits[:i + 1]) for i, s in enumerate(splits)]

    return splits
