"""
Code to split a set of molecules

- map_scaffolds: mapping unique scaffolds to the list of all scaffolds
- random_split: splits data at random
- stratified_random_split: splits data in a stratified manner
- scaffold_split: splits molecules based on their scaffolds


Derek van Tilborg
Eindhoven University of Technology
Jan 2024
"""

from collections import defaultdict
from rdkit import Chem
from typing import Union
from cheminformatics.utils import get_scaffold
import numpy as np


def map_scaffolds(smiles: list[str], scaffold_type: str = 'bemis_murcko') -> (list, dict[str, list[int]]):
    """ Find which molecules share the same scaffold

    :param mols: RDKit mol objects, e.g., as obtained through smiles_to_mols()
    :return: scaffolds, dict of unique scaffolds and which molecules (indices) share them -> {'c1ccccc1': [0, 12, 47]}
    """

    scaffolds = []
    for smi in smiles:

        mol = Chem.MolFromSmiles(smi)
        scaffold = get_scaffold(mol, scaffold_type=scaffold_type)
        scaff_smi = Chem.MolToSmiles(scaffold)
        scaffolds.append(scaff_smi)

    uniques = defaultdict(list)
    for i, s in enumerate(scaffolds):
        uniques[s].append(i)

    return scaffolds, uniques


def random_split(x: Union[int, list, np.ndarray], ratio: float = 0.2, seed: int = 42) -> (np.ndarray, np.ndarray):
    """ Random split data into a train and test split

    :param x: int representing the length of the data or iterable of the data itself to split
    :param ratio: test split ratio (default = 0.2, splits off 20% of the data into a test set)
    :param seed: random seed (default = 42)
    :return: train indices, test indices
    """
    n = x if type(x) is int else len(x)
    rng = np.random.default_rng(seed=seed)
    rand_idx = np.arange(n)
    rng.shuffle(rand_idx)

    test_idx = rand_idx[:round(n*ratio)]
    train_idx = rand_idx[round(n*ratio):]

    return train_idx, test_idx


def stratified_random_split(y: np.ndarray, ratio: float = 0.2, seed: int = 42) -> (np.ndarray, np.ndarray):
    """ Stratified random split data into a train and test split

    :param y: class vector used to stratify the data
    :param ratio: test split ratio (default = 0.2, splits off 20% of the data into a test set)
    :param seed: random seed (default = 42)
    :return: train indices, test indices
    """
    train_idx, test_idx = [], []

    # for every class in the class vector, split into train and test
    for c in set(y):
        y_c = np.where(y == c)[0]
        train_idx_y_c, test_idx_y_c = random_split(y_c, ratio=ratio, seed=seed)

        train_idx.append(y_c[train_idx_y_c])
        test_idx.append(y_c[test_idx_y_c])

    return np.concatenate(train_idx), np.concatenate(test_idx)


def scaffold_split(smiles: list[str], ratio: float = 0.2, scaffold_type: str = 'bemis_murcko',  seed: int = 42) -> \
        (np.ndarray, np.ndarray):
    """ Generates a random split based on Bismurcko scaffolds. Tries to deal with large set of scaffolds (sets
    containing >1% of the total number of scaffolds) by distributing those first and the smaller sets second.

    :param smiles: list of SMILES string
    :param ratio: test split ratio (default = 0.2, splits off a maximum of 20% of the data into a test set). Exact size
    of the split depends on scaffold set sizes.
    :param seed: random seed (default = 42)
    :return: train indices, test indices
    """
    rng = np.random.default_rng(seed=seed)
    testsetsize = round(len(smiles) * ratio)

    # Get scaffolds
    scaffolds, scaff_map = map_scaffolds(smiles, scaffold_type=scaffold_type)

    # When a set of scaffolds contains more than 1% of the total number of scaffolds, consider it a big set
    bigsetsize = round(len(scaff_map) * 0.01)

    big_sets = []
    small_sets = []
    for i, (k, v) in enumerate(scaff_map.items()):
        if len(v) > bigsetsize:
            big_sets.append(v)
        else:
            small_sets.append(v)

    # randomly suffle both sets
    rand_idx = np.arange(len(big_sets))
    rng.shuffle(rand_idx)
    big_sets = [big_sets[i] for i in rand_idx]

    rand_idx = np.arange(len(small_sets))
    rng.shuffle(rand_idx)
    small_sets = [small_sets[i] for i in rand_idx]

    # 1. Distribute large sets between train and test
    test_mols = []
    for i in range(len(big_sets)):
        if len(test_mols) < testsetsize - bigsetsize:  # Check if we can accommodate another large set
            if rng.choice([True, False], p=[ratio, 1-ratio]):  # decide if this big set will go to train or test
                # add this big set to the test
                test_mols.extend(big_sets[-1])
                big_sets = big_sets[:-1]  # get rid of the set we just added

    # 2. Distribute small sets between train and test
    for i in range(len(small_sets)):
        if len(test_mols) < testsetsize:
            test_mols.extend(small_sets[-1])
            small_sets = small_sets[:-1]  # get rid of the set we just added

    # add together the remaining molecules. This is the train set
    train_mols = sum(big_sets + small_sets, [])

    # randomly suffle both sets again so the same scaffolds are not clumped together
    rand_idx = np.arange(len(train_mols))
    rng.shuffle(rand_idx)
    train_mols = np.array(train_mols)[rand_idx]

    rand_idx = np.arange(len(test_mols))
    rng.shuffle(rand_idx)
    test_mols = np.array(test_mols)[rand_idx]

    return train_mols, test_mols
