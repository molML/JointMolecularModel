
import multiprocessing_on_dill as mp
from cheminformatics.utils import smiles_to_mols
from rdkit.DataStructs import BulkTanimotoSimilarity
from cheminformatics.utils import get_scaffold
from cheminformatics.descriptors import mols_to_ecfp
from cheminformatics.fractionalFMCS import MCSSimilarity
from tqdm.auto import tqdm
import numpy as np


def bulk_substructure_similarity(mol, mols, symmetric: bool = False):

        def calc_sim(*args):
            MSC = MCSSimilarity()
            return MSC.calc_similarity(*args)

        args_list = [(mol, m, symmetric) for m in mols]

        # Create a pool of worker processes
        with mp.Pool() as pool:
            # Use starmap to map the worker function to the argument tuples
            results = pool.starmap(calc_sim, args_list)

        return results


def mean_tani_bulk(ecfps, train_ecfps):

        def calc_sim(*args):
            Ti = BulkTanimotoSimilarity(*args)
            return np.mean(Ti)

        args_list = [(ecfp, train_ecfps) for ecfp in ecfps]

        # Create a pool of worker processes
        with mp.Pool() as pool:
            # Use starmap to map the worker function to the argument tuples
            results = pool.starmap(calc_sim, args_list)

        return results


def tani_sim_to_train(smiles: list[str], train_smiles: list[str], scaffold: bool = False, radius: int = 2,
                 nbits: int = 2048):
    """ Calculate the mean Tanimoto similarity between every molecule and the full train set

    :param smiles: list of SMILES strings
    :param train_smiles: list of train SMILES
    :param scaffold: bool to toggle the use of cyclic_skeletons
    :param radius: ECFP radius
    :param nbits: ECFP nbits
    :return: list of mean Tanimoto similarities
    """

    # get the ecfps for all smiles strings
    mols = smiles_to_mols(smiles)
    if scaffold:
        mols = [get_scaffold(m, scaffold_type='cyclic_skeleton') for m in mols]
    ecfps = mols_to_ecfp(mols, radius=radius, nbits=nbits)

    # get the ecfps for the body of train smiles
    train_mols = smiles_to_mols(train_smiles)
    if scaffold:
        train_mols = [get_scaffold(m, scaffold_type='cyclic_skeleton') for m in train_mols]
    train_ecfps = mols_to_ecfp(train_mols, radius=radius, nbits=nbits)

    T = mean_tani_bulk(ecfps, train_ecfps)

    return np.array(T)


def substructure_sim_to_train(smiles: list[str], train_smiles: list[str], scaffold: bool = False,
                              symmetric: bool = False):
    """ Calculate the mean substructure similarity between every molecule and the full train set

    :param smiles: list of SMILES strings
    :param train_smiles: list of train SMILES
    :param scaffold: bool to toggle the use of cyclic_skeletons
    :param symmetric: toggles symmetric similarity (i.e. f(a, b) = f(b, a))
    :return: list of mean substructure similarities
    """

    # get the ecfps for all smiles strings
    mols = smiles_to_mols(smiles)
    if scaffold:
        mols = [get_scaffold(m, scaffold_type='cyclic_skeleton') for m in mols]

    # get the ecfps for the body of train smiles
    train_mols = smiles_to_mols(train_smiles)
    if scaffold:
        train_mols = [get_scaffold(m, scaffold_type='cyclic_skeleton') for m in train_mols]

    S = []
    for mol in tqdm(mols):
        Si = bulk_substructure_similarity(mol, train_mols, symmetric)
        S.append(np.mean(Si))

    return np.array(S)