
from rdkit import Chem
from collections import Counter
import numpy as np
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem import rdMolDescriptors


# Implement the following complexity measures
# 	1.	BertzCT: Captures the structural branching and connectivity.
# 	2.	Shannon Entropy of the molecular graph: Measures the structural disorder and diversity of the molecule.
# 	3.	Shannon Entropy of the SMILES string: Measures the structural disorder and diversity of SMILES.
# 	4.	Number of Rotatable Bonds: Indicates the molecule’s flexibility.
# 	5.	Number of Functional Groups: Assesses the chemical diversity and functional complexity.


MOLECULAR_PATTERN_SMARTS = {
    # chains and branching
    "Long_chain groups": "[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]",  # Aliphatic chains at-least 8 members long.
    # cyclic features
    "Bicyclic": "[$([*R2]([*R])([*R])([*R]))].[$([*R2]([*R])([*R])([*R]))]",  # Bicyclic compounds have 2 bridgehead atoms with 3 arms connecting the bridgehead atoms.
    "Ortho": "*-!:aa-!:*",  # Ortho-substituted ring
    "Meta": "*-!:aaa-!:*",  # Meta-substituted ring
    "Para": "*-!:aaaa-!:*",  # Para-substituted ring
    "Macrocycle groups": "[r;!r3;!r4;!r5;!r6;!r7]",
    "Spiro-ring center": "[X4;R2;r4,r5,r6](@[r4,r5,r6])(@[r4,r5,r6])(@[r4,r5,r6])@[r4,r5,r6]",  # rings size 4-6
    "CIS or TRANS double bond in a ring": "*/,\[R]=;@[R]/,\*",  # An isomeric SMARTS consisting of four atoms and three bonds.
    "Unfused benzene ring": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",  # To find a benzene ring which is not fused, we write a SMARTS of 6 aromatic carbons in a ring where each atom is only in one ring:
    "Multiple non-fused benzene rings": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1.[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
    "Fused benzene rings": "c12ccccc1cccc2",
    # Functional groups:
    # carbonyl
    "Carbonyl group": "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]",  # Hits either resonance structure
    "Aldehyde": "[CX3H1](=O)[#6]",  # -al
    "Amide": "[NX3][CX3](=[OX1])[#6]",  # -amide
    "Carbamate": "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",  # Hits carbamic esters, acids, and zwitterions
    "Carboxylate Ion": "[CX3](=O)[O-]",  # Hits conjugate bases of carboxylic, carbamic, and carbonic acids.
    "Carbonic Acid or Carbonic Ester": "[CX3](=[OX1])(O)O",  # Carbonic Acid, Carbonic Ester, or combination
    "Carboxylic acid": "[CX3](=O)[OX1H0-,OX2H1]",
    "Ester Also hits anhydrides": "[#6][CX3](=O)[OX2H0][#6]",  # won't hit formic anhydride.
    "Ketone": "[#6][CX3](=O)[#6]",  # -one
    # ether
    "Ether": "[OD2]([#6])[#6]",
    # hydrogen atoms
    "Mono-Hydrogenated Cation": "[+H]",  # Hits atoms that have a positive charge and exactly one attached hydrogen:  F[C+](F)[H]
    # amide
    "Amidinium": "[NX3][CX3]=[NX3+]",
    "Cyanamide": "[NX3][CX2]#[NX1]",
    # amine
    "Primary or secondary amine, not amide": "[NX3;H2,H1;!$(NC=O)]",  # Not ammonium ion (N must be 3-connected), not ammonia (H count can't be 3). Primary or secondary is specified by N's H-count (H2 &amp; H1 respectively).  Also note that "&amp;" (and) is the dafault opperator and is higher precedence that "," (or), which is higher precedence than ";" (and). Will hit cyanamides and thioamides
    "Enamine or Aniline Nitrogen": "[NX3][$(C=C),$(cc)]",
    # azo
    "Azole": "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]",  # 5 member aromatic heterocycle w/ 2double bonds. contains N &amp; another non C (N,O,S)  subclasses are furo-, thio-, pyrro-  (replace
    # hydrazine
    "Hydrazine H2NNH2": "[NX3][NX3]",
    # hydrazone
    "Hydrazone C=NNH2": "[NX3][NX2]=[*]",
    # imine
    "Substituted or un-substituted imine": "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
    "Iminium": "[NX3+]=[CX3]",
    # imide
    "Unsubstituted dicarboximide": "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
    "Substituted dicarboximide": "[CX3](=[OX1])[NX3H0]([#6])[CX3](=[OX1])",
    # nitrate
    "Nitrate group": "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",  # Also hits nitrate anion
    # nitrile
    "Nitrile": "[NX1]#[CX2]",
    # nitro
    "Nitro group": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",   #Hits both forms.
    # hydroxyl (includes alcohol, phenol)
    "Hydroxyl": "[OX2H]",
    "Enol": "[OX2H][#6X3]=[#6]",
    "Phenol": "[OX2H][cX3]:[c]",
    # thio groups (thio-, thi-, sulpho-, marcapto-)
    "Carbo-Thioester": "S([#6])[CX3](=O)[#6]",
    "Thiol, Sulfide or Disulfide Sulfur": "[SX2]",
    "Thioamide": "[NX3][CX3]=[SX1]",
    # sulfide
    "Sulfide": "[#16X2H0]",  # -alkylthio  Won't hit thiols. Hits disulfides.
    "Mono-sulfide": "[#16X2H0][!#16]",  # alkylthio- or alkoxy- Won't hit thiols. Won't hit disulfides.
    "Two Sulfides": "[#16X2H0][!#16].[#16X2H0][!#16]",  # Won't hit thiols. Won't hit mono-sulfides. Won't hit disulfides.
    "Sulfone": "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",  # Hits all sulfones, including heteroatom-substituted sulfones:  sulfonic acid, sulfonate, sulfuric acid mono- &amp; di- esters, sulfamic acid, sulfamate, sulfonamide... Hits Both Depiction Forms.
    "Sulfonamide": "[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]",  # (sulf drugs)  Won't hit sulfamic acid or sulfamate. Hits Both Depiction Forms.
    # sulfoxide
    "Sulfoxide": "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",  # ( sulfinyl, thionyl ) Analog of carbonyl where S replaces C. Hits all sulfoxides, including heteroatom-substituted sulfoxides, dialkylsulfoxides carbo-sulfoxides, sulfinate, sulfinic acids... Hits Both Depiction Forms. Won't hit sulfones.
    # halide (-halo -fluoro -chloro -bromo -iodo)
    # Halogen
    "Halogen": "[F,Cl,Br,I]",
    # Three_halides groups
    "Three_halides groups": "[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]",  # Hits SMILES that have three halides.
}


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


def calculate_rotatable_bonds(mol: Mol) -> int:
    """ Count the number of rotatable bonds """

    return rdMolDescriptors.CalcNumRotatableBonds(mol)


def match_molecular_patterns(mol: Mol) -> dict[str, int]:
    """ Look for the presence of 50 distinct molecular patterns in a molecule (both functional groups and structural
    features).

    return a dict of patterns counts: {'Aldehyde': 1, 'Sulfide': 0, 'Ortho': 0, 'Meta': 0, 'Para': 2, ...}

    :return: pattern occurence
    """

    fg_counts = {}
    for name, smarts in MOLECULAR_PATTERN_SMARTS.items():
        fg_counts[name] = len(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)))

    return fg_counts


def count_unique_patterns(mol: Mol) -> int:
    """ Count how many different molecular patterns are present in a molecule, as defined in MOLECULAR_PATTERN_SMARTS

    :param mol: rdkit mol
    :return: number of unique molecular patterns in the molecule
    """
    pattern_counts = match_molecular_patterns(mol)

    return sum([1 for i in pattern_counts.values() if i > 0])


def calculate_smiles_shannon_entropy(smiles: str) -> float:
    """ Calculate the Shannon entropy of a SMILES string

    :math:`H=\sum_{i=1}^{n}{p_i\log_2p_i}`

    :param smiles: SMILES string
    :return: Shannon Entropy
    """

    # Count the frequency of each character in the SMILES string
    char_counts = Counter(smiles)

    # Total number of elements
    N = len(smiles)

    # Calculate the probabilities
    probabilities = [count / N for count in char_counts.values()]

    # Calculate Shannon Entropy
    shannon_entropy = sum(p * -np.log2(p) for p in probabilities)

    return shannon_entropy

