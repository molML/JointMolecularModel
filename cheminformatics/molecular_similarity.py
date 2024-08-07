from cheminformatics.multiprocessing import tanimoto_matrix
from cheminformatics.descriptors import mols_to_ecfp
from cheminformatics.utils import smiles_to_mols
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from cheminformatics.multiprocessing import bulk_cats, bulk_mcsf
from tqdm.auto import tqdm
from cheminformatics.utils import get_scaffold


def tanimoto_matrix_from_smiles(smiles_a, smiles_b, radius: int = 2, nbits: int = 2048) -> np.ndarray:
    """ Compute the Tanimoto similarity between molecules in set A (rows) and set B (cols). A and B should be
    the same for a pair-wise matrix

    :param: smiles_a: list of SMILES strings (will be rows)
    :param: smiles_b: list of SMILES strings (will be columns)
    :param: radius: ECFP radius
    :param: nbits: ECFP bits
    :return: np array of shape (smiles_a x smiles_b)
    """

    mols_a = smiles_to_mols(smiles_a)
    ecfps_a = mols_to_ecfp(mols_a, radius=radius, nbits=nbits)

    mols_b = smiles_to_mols(smiles_b)
    ecfps_b = mols_to_ecfp(mols_b, radius=radius, nbits=nbits)

    T = tanimoto_matrix(ecfps_a, ecfps_b)

    return T


def applicability_domain_kNN(query_smiles: list[str], train_smiles: list[str], k: int = 10,
                             sim_cutoff: float = 0.25, radius: int = 2, nbits: int = 2048) -> list[bool]:
    """ Checks if query molecules are in the applicability domain given some training molecules.

    If N molecules in the training set have a Tanimoto similarity of Scutoff of higher with the query molecule,
    it is condidered to be inside the applicability domain, according to

        ADfingerprint{Nmin, Scutoff}.

    Wang et al., Chem. Res. Toxicol. 2020, 33, 1382−1388
    https://pubs.acs.org/doi/full/10.1021/acs.chemrestox.9b00498

    :param: query_smiles: list of SMILES strings that you want to check
    :param: train_smiles: list of SMILES strings of the training set
    :param: k: number of molecules in the train set that need to be close to the query mol (Nmin in the paper)
    :param: radius: ECFP radius
    :param: nbits: ECFP bits
    :param: sim_cutoff: how close do nmin molecules in the train set have to be to the query molecule
    :return: list of bools wheter or not a query molecule is inside of the applicability domain
    """

    T = tanimoto_matrix_from_smiles(query_smiles, train_smiles, radius=radius, nbits=nbits)

    in_applicabilitydomain = [(ti >= sim_cutoff).sum() >= k for ti in T]

    return in_applicabilitydomain


def applicability_domain_SDC(query_smiles: list[str], train_smiles: list[str], radius: int = 2,
                             nbits: int = 2048) -> np.ndarray:
    """ Computes the SDC, which is the sum of the distance-weighted contributions for a set of query molecules
    given a set of training molecules according to:

    :math:`SDC\ =\ \sum_{j=1}^{n}exp\left(\frac{-3(1- T_{ij})\ }{T_{ij}}\right)`

    Liu and Wallqvist, Journal of Chemical Information and Modeling 2019 59 (1), 181-189
    DOI: 10.1021/acs.jcim.8b00597

    :param: query_smiles: list of SMILES strings that you want to check
    :param: train_smiles: list of SMILES strings of the training set
    :param: radius: ECFP radius
    :param: nbits: ECFP bits
    :return: array of SDC values
    """

    # Compute tanimoto similarity matrix and turn it into a distance matrix
    T = tanimoto_matrix_from_smiles(query_smiles, train_smiles, radius=radius, nbits=nbits)
    Tdist = 1 - T

    sdc = np.exp((-3 * Tdist) / (1 - Tdist)).sum(axis=1)

    return sdc


def mean_cosine_cats_to_train(smiles: list[str], train_smiles: list[str]):
    """ Calculate the mean Tanimoto similarity between every molecule and the full train set

    :param smiles: list of SMILES strings
    :param train_smiles: list of train SMILES
    :param scaffold: bool to toggle the use of cyclic_skeletons
    :param radius: ECFP radius
    :param nbits: ECFP nbits
    :return: list of mean Tanimoto similarities
    """

    # get the cats for all smiles strings
    all_cats = bulk_cats(smiles_to_mols(smiles))
    train_cats = bulk_cats(smiles_to_mols(train_smiles))

    # compute cosine sim
    S = []
    for cats_i in all_cats:
        s_i = cosine_similarity(np.array([cats_i]), train_cats)
        S.append(np.mean(s_i))

    return np.array(S)


def mcsf_to_train(smiles: list[str], train_smiles: list[str], scaffold: bool = False,
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
        Si = bulk_mcsf(mol, train_mols, symmetric)
        S.append(np.mean(Si))

    return np.array(S)


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
