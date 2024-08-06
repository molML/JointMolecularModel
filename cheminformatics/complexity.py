
from rdkit import Chem
from collections import Counter
import numpy as np
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem import rdMolDescriptors


# Implement the following complexity measures
# 	1.	BertzCT: Captures the structural branching and connectivity.
# 	2.	Topological Polar Surface Area (TPSA): Reflects the molecule’s polarity and surface properties.
# 	3.	Number of Rotatable Bonds: Indicates the molecule’s flexibility.
# 	4.	Shannon Entropy: Measures the structural disorder and diversity.
# 	5.	Number of Functional Groups: Assesses the chemical diversity and functional complexity.


def calculate_shannon_entropy(mol: Mol) -> float:
    """ Compute the shannon entropy of a molecular graph.

    The Shannon entropy I for a molecule with N elements is:

    :math:`I\ =\ Nlog_2\ N\ -\ \sum_(i=1)^n\of\beginN_i\ log_2\ N_i\ )\`

    where n is the number of different sets of elements and Ni is the number of elements in the ith set of elements.

    Bonchev, D., Kamenski, D., & Kamenska, V. (1976). Symmetry and information content of chemical structures.
    Bulletin of Mathematical Biology, 38(2), 119-133.

    Important caveat: homonuclear molecules (e.g. a buckyball) will yield a Shannon entropy of 0

    :param mol: RDKit molecule
    :return: Shannon Entropy of the molecular graph
    """

    # Get the symbol of each atom (element)
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]

    # Calculate frequency of each element
    element_counts = Counter(elements)

    # Total number of elements
    N = len(elements)

    # Calculate Nlog2(N)
    entropy_part1 = N * np.log2(N)

    # Calculate the sum of Ni * log2(Ni) for each distinct element
    entropy_part2 = sum(Ni * np.log2(Ni) for Ni in element_counts.values())

    # Calculate the entropy
    entropy = entropy_part1 - entropy_part2

    return entropy


def calculate_bertz_complexity(mol: Mol, **kwargs) -> float:
    """ Compute the Bertz complexity, which is a measure of molecular complexity """

    return BertzCT(mol, **kwargs)


def calculate_tpsa(mol: Mol) -> float:
    """ Compute the total polar surface area """

    return rdMolDescriptors.CalcTPSA(mol)


def calculate_rotatable_bonds(mol: Mol) -> float:
    """ Count the number of rotatable bonds """

    return rdMolDescriptors.CalcNumRotatableBonds(mol)
