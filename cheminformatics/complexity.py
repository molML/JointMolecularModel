"""

These are several metric used to approximate the complexity of a molecule from different angles.

# 	1.	BertzCT: Captures structural branching and connectivity.
#   2.  Böttcher Complexity: https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00723
# 	2.	Shannon Entropy of the molecular graph: Measures the structural disorder and diversity of the molecule.
# 	3.	Shannon Entropy of the SMILES string: Measures the structural disorder and diversity of SMILES.
# 	5.	Number of structural motifs: Assesses the chemical diversity and functional complexity.

In this script I have reimplemented code from
- Zach Pearson: https://github.com/boskovicgroup/bottchercomplexity/tree/main
- Nadine Schneider & Peter Ertl: https://github.com/rdkit/rdkit/blob/master/Contrib/ChiralPairs/ChiralDescriptors.py

Derek van Tilborg
Eindhoven University of Technology
July 2024
"""

import math
from collections import Counter, defaultdict
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.GraphDescriptors import BertzCT
from cheminformatics.encoding import smiles_tokenizer


MOLECULAR_PATTERN_SMARTS = {
    # chains and branching
    "Long_chain groups": "[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]",  # Aliphatic chains at-least 8 members long.
    # cyclic features
    "Bicyclic": "[$([*R2]([*R])([*R])([*R]))].[$([*R2]([*R])([*R])([*R]))]",
    # Bicyclic compounds have 2 bridgehead atoms with 3 arms connecting the bridgehead atoms.
    "Ortho": "*-!:aa-!:*",  # Ortho-substituted ring
    "Meta": "*-!:aaa-!:*",  # Meta-substituted ring
    "Para": "*-!:aaaa-!:*",  # Para-substituted ring
    "Macrocycle groups": "[r;!r3;!r4;!r5;!r6;!r7]",
    "Spiro-ring center": "[X4;R2;r4,r5,r6](@[r4,r5,r6])(@[r4,r5,r6])(@[r4,r5,r6])@[r4,r5,r6]",  # rings size 4-6
    "CIS or TRANS double bond in a ring": "*/,\[R]=;@[R]/,\*",
    # An isomeric SMARTS consisting of four atoms and three bonds.
    "Unfused benzene ring": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
    # To find a benzene ring which is not fused, we write a SMARTS of 6 aromatic carbons in a ring where each atom is only in one ring:
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
    "Mono-Hydrogenated Cation": "[+H]",
    # Hits atoms that have a positive charge and exactly one attached hydrogen:  F[C+](F)[H]
    # amide
    "Amidinium": "[NX3][CX3]=[NX3+]",
    "Cyanamide": "[NX3][CX2]#[NX1]",
    # amine
    "Primary or secondary amine, not amide": "[NX3;H2,H1;!$(NC=O)]",
    # Not ammonium ion (N must be 3-connected), not ammonia (H count can't be 3). Primary or secondary is specified by N's H-count (H2 &amp; H1 respectively).  Also note that "&amp;" (and) is the dafault opperator and is higher precedence that "," (or), which is higher precedence than ";" (and). Will hit cyanamides and thioamides
    "Enamine or Aniline Nitrogen": "[NX3][$(C=C),$(cc)]",
    # azo
    "Azole": "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]",
    # 5 member aromatic heterocycle w/ 2double bonds. contains N &amp; another non C (N,O,S)  subclasses are furo-, thio-, pyrro-  (replace
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
    "Nitro group": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",  # Hits both forms.
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
    "Two Sulfides": "[#16X2H0][!#16].[#16X2H0][!#16]",
    # Won't hit thiols. Won't hit mono-sulfides. Won't hit disulfides.
    "Sulfone": "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
    # Hits all sulfones, including heteroatom-substituted sulfones:  sulfonic acid, sulfonate, sulfuric acid mono- &amp; di- esters, sulfamic acid, sulfamate, sulfonamide... Hits Both Depiction Forms.
    "Sulfonamide": "[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]",
    # (sulf drugs)  Won't hit sulfamic acid or sulfamate. Hits Both Depiction Forms.
    # sulfoxide
    "Sulfoxide": "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",
    # ( sulfinyl, thionyl ) Analog of carbonyl where S replaces C. Hits all sulfoxides, including heteroatom-substituted sulfoxides, dialkylsulfoxides carbo-sulfoxides, sulfinate, sulfinic acids... Hits Both Depiction Forms. Won't hit sulfones.
    # halide (-halo -fluoro -chloro -bromo -iodo)
    # Halogen
    "Halogen": "[F,Cl,Br,I]",
    # Three_halides groups
    "Three_halides groups": "[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]",  # Hits SMILES that have three halides.
}


def molecular_complexity(smiles: str) -> dict[str, float]:
    """

    :param smiles: SMILES string
    :return: a dictionary with several measures of complexity
    """
    mol = Chem.MolFromSmiles(smiles)

    complexity = {"bertz": calculate_bertz_complexity(mol),
                  "bottcher": calculate_bottcher_complexity(mol),
                  "molecule_entropy": calculate_molecular_shannon_entropy(mol),
                  "smiles_entropy": calculate_smiles_shannon_entropy(smiles),
                  "motifs": count_unique_motifs(mol)}

    return complexity


def calculate_molecular_shannon_entropy(mol: Mol) -> float:
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


def calculate_smiles_shannon_entropy(smiles: str) -> float:
    """ Calculate the Shannon entropy of a SMILES string by its tokens. Start, end-of-sequence, and padding tokens are
    not considered.

    :math:`H=\sum_{i=1}^{n}{p_i\log_2p_i}`

    :param smiles: SMILES string
    :return: Shannon Entropy
    """

    tokens = smiles_tokenizer(smiles)

    # Count the frequency of each token in the SMILES string
    char_counts = Counter(tokens)

    # Total number of tokens in the SMILES string
    N = len(tokens)

    # Calculate the probabilities
    probabilities = [count / N for count in char_counts.values()]

    # Calculate Shannon Entropy
    shannon_entropy = sum(p * -np.log2(p) for p in probabilities)

    return shannon_entropy


def calculate_bertz_complexity(mol: Mol, **kwargs) -> float:
    """ Compute the Bertz complexity, which is a measure of molecular complexity """

    return BertzCT(mol, **kwargs)


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


def count_unique_motifs(mol: Mol) -> int:
    """ Count how many different molecular patterns are present in a molecule, as defined in MOLECULAR_PATTERN_SMARTS

    :param mol: rdkit mol
    :return: number of unique molecular patterns in the molecule
    """
    pattern_counts = match_molecular_patterns(mol)

    return sum([1 for i in pattern_counts.values() if i > 0])


def calculate_bottcher_complexity(mol: Mol, debug: bool = False) -> float:
    """

    Böttcher complexity according to Zach Pearsons implementation
    https://github.com/boskovicgroup/bottchercomplexity/tree/main

    This complexity measure consists of the sum of several atom-wise parts

    :param mol: RDkit mol
    :param debug: bool to toggle some print statements
    :return: Böttcher complexity
    """

    complexity = 0
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    atoms = mol.GetAtoms()
    atom_stereo_classes = []
    atoms_corrected_for_symmetry = []
    for atom in atoms:
        if atom.GetProp('_CIPRank') in atom_stereo_classes:
            continue
        else:
            atoms_corrected_for_symmetry.append(atom)
            atom_stereo_classes.append(atom.GetProp('_CIPRank'))
    for atom in atoms_corrected_for_symmetry:
        d = _GetChemicalNonequivs(atom, mol)
        e = _GetBottcherLocalDiversity(atom)
        s = _GetNumIsomericPossibilities(atom)
        V = _GetNumValenceElectrons(atom)
        b = _GetBottcherBondIndex(atom)
        if debug:
            print(f'Atom: {atom.GetSymbol()}')
            print('\tSymmetry Class: ' + str(atom.GetProp('_CIPRank')))
            print('\tCurrent Parameter Values:')
            print('\t\td_sub_i: ' + str(d))
            print('\t\te_sub_i: ' + str(e))
            print('\t\ts_sub_i: ' + str(s))
            print('\t\tV_sub_i: ' + str(V))
            print('\t\tb_sub_i: ' + str(b))
        complexity += d * e * s * math.log(V * b, 2)
    if debug:
        print('Current Complexity Score: ' + str(complexity))
        return None

    return complexity


def _determineAtomSubstituents(atomID, mol, distanceMatrix, verbose=False):
    """

    Copyright (c) 2017, Novartis Institutes for BioMedical Research Inc.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above
          copyright notice, this list of conditions and the following
          disclaimer in the documentation and/or other materials provided
          with the distribution.
        * Neither the name of Novartis Institutes for BioMedical Research Inc.
          nor the names of its contributors may be used to endorse or promote
          products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    Created by Nadine Schneider & Peter Ertl, July 2017

    https://github.com/rdkit/rdkit/blob/master/Contrib/ChiralPairs/ChiralDescriptors.py
    """

    atomPaths = distanceMatrix[atomID]
    # determine the direct neighbors of the atom
    neighbors = [n for n, i in enumerate(atomPaths) if i == 1]
    # store the ids of the neighbors (substituents)
    subs = defaultdict(list)
    # track in how many substituents an atom is involved (can happen in rings)
    sharedNeighbors = defaultdict(int)
    # determine the max path length for each substituent
    maxShell = defaultdict(int)
    for n in neighbors:
        subs[n].append(n)
        sharedNeighbors[n] += 1
        maxShell[n] = 0
    # second shell of neighbors
    mindist = 2
    # max distance from atom
    maxdist = int(np.max(atomPaths))
    for d in range(mindist, maxdist + 1):
        if verbose:
            print("Shell: ", d)
        newShell = [n for n, i in enumerate(atomPaths) if i == d]
        for aidx in newShell:
            if verbose:
                print("Atom ", aidx, " in shell ", d)
            atom = mol.GetAtomWithIdx(aidx)
            # find neighbors of the current atom that are part of the substituent already
            for n in atom.GetNeighbors():
                nidx = n.GetIdx()
                for k, v in subs.items():
                    # is the neighbor in the substituent and is not in the same shell as the current atom
                    # and we haven't added the current atom already then put it in the correct substituent list
                    if nidx in v and nidx not in newShell and aidx not in v:
                        subs[k].append(aidx)
                        sharedNeighbors[aidx] += 1
                        maxShell[k] = d
                        if verbose:
                            print("Atom ", aidx, " assigned to ", nidx)
    if verbose:
        print(subs)
        print(sharedNeighbors)

    return subs, sharedNeighbors, maxShell


def _GetChemicalNonequivs(atom, mol):
    """ D

    Current failures: Does not distinguish between cyclopentyl and pentyl (etc.)
                      and so unfairly underestimates complexity.

    :param atom:
    :param themol:
    :return:
    """

    num_unique_substituents = 0
    substituents = [[], [], [], []]
    for item, key in enumerate(_determineAtomSubstituents(atom.GetIdx(), mol, Chem.GetDistanceMatrix(mol))[0]):
        for subatom in _determineAtomSubstituents(atom.GetIdx(), mol, Chem.GetDistanceMatrix(mol))[0][key]:
            substituents[item].append(mol.GetAtomWithIdx(subatom).GetSymbol())
            num_unique_substituents = len(
                set(tuple(tuple(substituent) for substituent in substituents if substituent)))

    return num_unique_substituents


def _GetBottcherLocalDiversity(atom):
    """ E

    The number of different non-hydrogen elements or isotopes (including deuterium
    and tritium) in the atom's microenvironment.

    CH4 - the carbon has e_i of 1
    Carbonyl carbon of an amide e.g. CC(=O)N e_i = 3
        while N and O have e_i = 2
    """

    neighbors = []
    for neighbor in atom.GetNeighbors():
        neighbors.append(str(neighbor.GetSymbol()))
    if atom.GetSymbol() in set(neighbors):
        return len(set(neighbors))
    else:
        return len(set(neighbors)) + 1


def _GetNumIsomericPossibilities(atom):
    """ S

    RDKit marks atoms where there is potential for isomerization with a tag
    called _CIPCode. If it exists for an atom, note that S = 2, otherwise 1.
    """

    try:
        if (atom.GetProp('_CIPCode')):
            return 2
    except:
        return 1


def _GetNumValenceElectrons(atom):
    """ V

    The number of valence electrons the atom would have if it were unbonded and
    neutral
    """

    valence = {1: ['H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'],  # Alkali Metals
               2: ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'],  # Alkali Earth Metals
               # transition metals???
               3: ['B', 'Al', 'Ga', 'In', 'Tl', 'Nh'],  #
               4: ['C', 'Si', 'Ge', 'Sn', 'Pb', 'Fl'],
               5: ['N', 'P', 'As', 'Sb', 'Bi', 'Mc'],  # Pnictogens
               6: ['O', 'S', 'Se', 'Te', 'Po', 'Lv'],  # Chalcogens
               7: ['F', 'Cl', 'Br', 'I', 'At', 'Ts'],  # Halogens
               8: ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Og']}  # Noble Gases
    for k in valence:
        if atom.GetSymbol() in valence[k]:
            return k
    return 0


def _GetBottcherBondIndex(atom):
    """ B

    Represents the total number of bonds to other atoms with V_i*b_i > 1, so
    basically bonds to atoms other than Hydrogen

    Here we can leverage the fact that RDKit does not even report Hydrogens by
    default to simply loop over the bonds. We will have to account for molecules
    that have hydrogens turned on before we can submit this code as a patch
    though.
    """

    b_sub_i_ranking = 0
    bonds = []
    for bond in atom.GetBonds():
        bonds.append(str(bond.GetBondType()))
    for bond in bonds:
        if bond == 'SINGLE':
            b_sub_i_ranking += 1
        if bond == 'DOUBLE':
            b_sub_i_ranking += 2
        if bond == 'TRIPLE':
            b_sub_i_ranking += 3
    if 'AROMATIC' in bonds:
        # This list can be expanded as errors arise.
        if atom.GetSymbol() == 'C':
            b_sub_i_ranking += 3
        elif atom.GetSymbol() == 'N':
            b_sub_i_ranking += 2
        elif atom.GetSymbol() == 'O':  # I expanded this to O
            b_sub_i_ranking += 2

    if b_sub_i_ranking == 0:
        b_sub_i_ranking += 1

    return b_sub_i_ranking
