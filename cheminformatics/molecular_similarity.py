import numpy as np
from tqdm.auto import tqdm
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
import torch
from sklearn.metrics.pairwise import cosine_similarity
from cheminformatics.multiprocessing import tanimoto_matrix, bulk_cats, bulk_mcsf
from cheminformatics.descriptors import mols_to_ecfp
from cheminformatics.utils import smiles_to_mols, get_scaffold


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


def tani_sim_to_train(smiles: list[str], train_smiles: list[str], scaffold: bool = False, radius: int = 2,
                      nbits: int = 2048, mol_library: dict = None):
    """ Calculate the mean Tanimoto similarity between every molecule and the full train set

    :param smiles: list of SMILES strings
    :param train_smiles: list of train SMILES
    :param scaffold: bool to toggle the use of cyclic_skeletons
    :param radius: ECFP radius
    :param nbits: ECFP nbits
    :param mol_library" dict of {smiles: mol} of premade mol objects
    :return: list of mean Tanimoto similarities
    """
    if not mol_library:
        mol_library = {smi: Chem.MolFromSmiles(smi) for smi in tqdm(smiles)}

    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)

    print('\t\tMaking a fingerprint library')
    if scaffold:
        fp_library = {smi: mfpgen.GetFingerprint(get_scaffold(mol_library[smi], scaffold_type='cyclic_skeleton')) for smi in tqdm(smiles)}
    else:
        fp_library = {smi: mfpgen.GetFingerprint(mol_library[smi]) for smi in tqdm(smiles)}

    print('\t\tComputing Tanimoto similarities between ECFPs')
    T = tanimoto_matrix([fp_library[smi_i] for smi_i in smiles], [fp_library[smi_j] for smi_j in train_smiles], take_mean=True)

    return T


def mcsf_to_train(smiles: list[str], train_smiles: list[str], scaffold: bool = False, symmetric: bool = False,
                  mol_library: dict = None):
    """ Calculate the mean substructure similarity between every molecule and the full train set

    :param smiles: list of SMILES strings
    :param train_smiles: list of train SMILES
    :param scaffold: bool to toggle the use of cyclic_skeletons
    :param symmetric: toggles symmetric similarity (i.e. f(a, b) = f(b, a))
    :param mol_library" dict of {smiles: mol} of premade mol objects
    :return: list of mean substructure similarities
    """

    if not mol_library:
        mol_library = {smi: Chem.MolFromSmiles(smi) for smi in tqdm(smiles)}
    if scaffold:
        mol_library = {smi: get_scaffold(mol_library[smi], scaffold_type='cyclic_skeleton') for smi in tqdm(smiles)}

    S = []
    train_mols = [mol_library[smi_j] for smi_j in train_smiles]
    for smi_i in tqdm(smiles):
        Si = bulk_mcsf(mol_library[smi_i], train_mols, symmetric)
        S.append(np.mean(Si))

    return np.array(S)


def mean_cosine_cats_to_train(smiles: list[str], train_smiles: list[str], mol_library: dict = None):
    """ Calculate the mean Tanimoto similarity between every molecule and the full train set

    :param smiles: list of SMILES strings
    :param train_smiles: list of train SMILES
    :param mol_library" dict of {smiles: mol} of premade mol objects
    :return: list of mean CATs cosine similarities
    """

    if not mol_library:
        mol_library = {smi: Chem.MolFromSmiles(smi) for smi in tqdm(smiles)}

    cats_library = bulk_cats([mol_library[smi_i] for smi_i in smiles])
    cats_library = {smi: cat for smi, cat in zip(smiles, cats_library)}

    # get the cats for all smiles strings
    all_cats = [cats_library[smi_i] for smi_i in smiles]
    train_cats = [cats_library[smi_j] for smi_j in train_smiles]

    del cats_library

    # compute cosine sim
    S = []
    for cats_i in tqdm(all_cats):
        s_i = cosine_similarity(np.array([cats_i]), train_cats)
        S.append(np.mean(s_i))

    return np.array(S)


def compute_z_distance_to_train(model, dataset, train_dataset) -> torch.Tensor:
    """ Compute the mean mahalanobis distance for every sample in a dataset to all samples in the train dataset """

    train_z, _ = model.get_z(train_dataset)
    other_z, _ = model.get_z(dataset)

    return mahalanobis_mean_distance(train_z.cpu(), other_z.cpu())


def mahalanobis_mean_distance(A: torch.Tensor, B: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """ Computes the mean Mahalanobis distance from every sample in B to all samples in A.

    :param A: tensor of (samples, features)
    :param B: tensor of (samples, features)
    :param epsilon: small float
    :return: tensor (samples A)
    """

    # Compute the covariance matrix of A
    cov_matrix = compute_covariance_matrix(A)

    # Add regularization to the covariance matrix
    cov_matrix += epsilon * torch.eye(cov_matrix.size(0), device=cov_matrix.device)

    # Compute the inverse of the regularized covariance matrix
    cov_inv = torch.inverse(cov_matrix)

    # Initialize a tensor to store the mean distances
    mean_distances = torch.zeros(B.size(0), device=B.device)

    # Compute mean Mahalanobis distance for each sample in B
    for i, b in enumerate(B):
        # Compute Mahalanobis distance from b to all samples in A
        deltas = A - b  # Shape: (m, n)
        distances = torch.sqrt(torch.einsum('ij,jk,ik->i', deltas, cov_inv, deltas))  # Shape: (m,)
        mean_distances[i] = distances.mean()  # Mean distance for the current sample in B

    return mean_distances


def compute_covariance_matrix(x: torch.Tensor) -> torch.Tensor:
    """ Compute the covariance matrix from x (samples, features) """

    mean = x.mean(dim=0, keepdim=True)
    x_centered = x - mean
    n = x.size(0)
    cov = x_centered.T @ x_centered / (n - 1)

    return cov
