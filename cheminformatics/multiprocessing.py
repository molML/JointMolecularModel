
import multiprocessing_on_dill as mp
from cheminformatics.utils import smiles_to_mols
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.Chem.rdchem import Mol
from cheminformatics.utils import get_scaffold
from cheminformatics.descriptors import mols_to_ecfp, cats
from cheminformatics.fractionalFMCS import MCSSimilarity
import numpy as np


def bulk_substructure_similarity(mol: Mol, mols: list[Mol], symmetric: bool = False):

        def calc_sim(*args):
            MSC = MCSSimilarity()
            return MSC.calc_similarity(*args)

        args_list = [(mol, m, symmetric) for m in mols]

        # Create a pool of worker processes
        with mp.Pool() as pool:
            # Use starmap to map the worker function to the argument tuples
            results = pool.starmap(calc_sim, args_list)

        return results


def bulk_cats(mols: list[Mol] | Mol, to_array: bool = True) -> np.ndarray:

        if type(mols) is not list:
            mols = [mols]

        # Create a pool of worker processes
        with mp.Pool() as pool:
            # Use starmap to map the worker function to the argument tuples
            results = pool.map(cats, mols)

        if to_array:
            return np.array(results)
        return results


def tanimoto_matrix(*fingerprints, dtype=np.float16) -> np.ndarray:
    """ Compute the pair-wise Tanimoto coefficient. You can either give two sets of fingerprints, in which case the
     resulting matrix consists of similarity between molecules in set A (rows) and set B (cols), or give one set of
     fingerprints, in which it will be a symmetrical matrix (diagonal will be 1).

    :param fingerprints: list of fingerprints, you can supply either one or two of them.
    :param dtype: numpy dtype (default = np.float16)
    :return: Tanimoto similarity matrix
    """

    def calc_sim(*args):
        Ti = BulkTanimotoSimilarity(*args)
        return np.array(Ti, dtype=dtype)

    if len(fingerprints) == 2:
        ecfps_a, ecfps_b = fingerprints
    else:
        ecfps_a = ecfps_b = fingerprints[0]

    # Fill the whole Tanimoto matrix. Experimenting with just filling the upper triangle and then mirroring it out for
    # symmetric cases didn't seem to be faster because you have to reconstruct the triangle with a for loop
    args_list = [(ecfp, ecfps_b) for ecfp in ecfps_a]
    with mp.Pool() as pool:
        T = pool.starmap(calc_sim, args_list)

    return np.array(T, dtype=dtype)


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

    T = tanimoto_matrix(ecfps, train_ecfps)
    mean_tani_sim_to_train = np.mean(T, 1)

    return mean_tani_sim_to_train
