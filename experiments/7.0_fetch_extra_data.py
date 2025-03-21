#####
# Function to perform outlier analysis and compute descriptive metrics
# Francesca Grisoni, TU/e April 2021
#####

import os
import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter
from cheminformatics.cleaning import clean_mols
from constants import ROOTDIR


def curate_exp(df, outlier_test=True, alpha=0.05, convert_to_log=False, rm_high_std=True):
    """

    :param df:
    :param outlier_test:
    :param alpha:
    :param convert_to_log:
    :return:
    """

    # init
    import pandas as pd
    df_curated = pd.DataFrame()

    # list of unique smiles to be used to merge multiple entries
    unique_smiles = df.CuratedSmiles.unique()
    print(str(len(df.CuratedSmiles.unique())) + ' unique molecules out of ' + str(len(df.CuratedSmiles)) + ' entries.')

    # iterates over the unique SMILES and retrieves the important information
    for smi in unique_smiles:
        # retrieves the data
        data = df.loc[df['CuratedSmiles'] == smi]

        # === experimental values curation
        exp_values = data["value"]  # retrieves the experimental values
        exp_values = exp_values.values.astype(float)  # overwrites and converts into np float array
        mean, std, sem, num_outliers, num_values, new_values, outliers_index = curate_values(exp_values,
                                                                                             outlier_test=outlier_test,
                                                                                             alpha=alpha,
                                                                                             convert_to_log=convert_to_log)
        # === keeps track of the original values (endpoint type, ChEMBL doc ID)
        chembl_ids, chembl_docs_ids, standard_types = concatenate_string_info(data, outliers_index)

        # === generates the new table
        df_curated = df_curated._append({'smiles': smi, 'exp_mean': mean, 'exp_std': std, 'exp_sem': sem,
                                        'no.entries': num_values,
                                        'no.outliers': num_outliers,
                                        'standard_types': standard_types,
                                        'chembl_id': chembl_ids,
                                        'chembl_id_doc': chembl_docs_ids},
                                       ignore_index=True)

    # If the std is higher than 1 log unit, discard the molecules
    if rm_high_std:
        df_curated.drop(df_curated.loc[np.log10(df_curated['exp_std']) >= 1].index, inplace=True)

    return df_curated


def curate_values(values, outlier_test=True, alpha=0.05, convert_to_log=False):
    # converts to -log if the option is true.
    if convert_to_log is True:
        values = -np.log10(np.divide(values, 1000000000))  # converts to molar units and computes the negative log.

    # init
    new_values = values.copy()  # copy for potential deletion of outliers
    num_outliers = 'na'  # will be overwritten in case the test is performed
    outliers_index = []  # will be overwritten in case of outliers

    # performs Dixon's Q
    if outlier_test:
        outliers = dixon_test(values, left=True, right=True, alpha=alpha)

        if outliers:
            outliers_index = np.where(values == outliers)
            new_values = np.delete(values, outliers_index)  # overwrites new_values in case of outl.
            num_outliers = outliers.size
        else:
            num_outliers = 0

    # compute stats
    mean, std, sem, num_values = compute_stats(new_values)

    return mean, std, sem, num_outliers, num_values, new_values, outliers_index


# ========== Dixon's Q test ==================
def _InitLookUp(alpha=0.05):
    """ Initialize the dictionary of the look up table based on the confidence level. The tabulated data are from
    Rorabacher, D.B.(1991) Analytical Chemistry 63(2), 139–46.

    :param alpha: confidence interval (double), accepted values: 0.1, 0.05, 0.01
    :return:
    """
    if alpha == 0.10:
        q_tab = [0.941, 0.765, 0.642, 0.56, 0.507, 0.468, 0.437,
                 0.412, 0.392, 0.376, 0.361, 0.349, 0.338, 0.329,
                 0.32, 0.313, 0.306, 0.3, 0.295, 0.29, 0.285, 0.281,
                 0.277, 0.273, 0.269, 0.266, 0.263, 0.26]
    elif alpha == 0.05:
        q_tab = [0.97, 0.829, 0.71, 0.625, 0.568, 0.526, 0.493, 0.466,
                 0.444, 0.426, 0.41, 0.396, 0.384, 0.374, 0.365, 0.356,
                 0.349, 0.342, 0.337, 0.331, 0.326, 0.321, 0.317, 0.312,
                 0.308, 0.305, 0.301, 0.29]
    elif alpha == 0.01:
        q_tab = [0.994, 0.926, 0.821, 0.74, 0.68, 0.634, 0.598, 0.568,
                 0.542, 0.522, 0.503, 0.488, 0.475, 0.463, 0.452, 0.442,
                 0.433, 0.425, 0.418, 0.411, 0.404, 0.399, 0.393, 0.388,
                 0.384, 0.38, 0.376, 0.372]
    else:
        print("Input alpha value not available")
        q_tab = []

    Q = {n: q for n, q in zip(range(3, len(q_tab) + 1), q_tab)}
    return Q


def dixon_test(data, left=True, right=True, alpha=0.05):
    """
    Keyword arguments:
        data = A ordered or unordered list of data points (int or float).
        left = Q-test of minimum value in the ordered list if True.
        right = Q-test of maximum value in the ordered list if True.
        q_dict = A dictionary of Q-values for a given confidence level,
            where the dict. keys are sample sizes N, and the associated values
            are the corresponding critical Q values. E.g.,
            {3: 0.97, 4: 0.829, 5: 0.71, 6: 0.625, ...}

    Returns a list of two values for the outliers, or None.
    E.g.,
       for [1,1,1] -> [None, None]
       for [5,1,1] -> [None, 5]
       for [5,1,5] -> [1, None]
    # inspired by https://sebastianraschka.com/Articles/2014_dixon_test.html

    """
    # retrieves the tabulated data
    q_dict = _InitLookUp(alpha=0.05)

    # preliminary controls
    assert (left or right), 'At least one of the variables, `left` or `right`, must be True'

    if 3 <= len(data) <= max(q_dict.keys()):  # checks num values for stat. significance
        # sorts data to perform the analysis on the extreme values
        sdata = sorted(data)
        Q_mindiff, Q_maxdiff = (0, 0), (0, 0)

        if left:
            Q_min = (sdata[1] - sdata[0])
            try:
                Q_min /= (sdata[-1] - sdata[0])
                Q_mindiff = (Q_min - q_dict[len(data)], sdata[0])
            except ZeroDivisionError:
                pass
        if right:
            Q_max = abs((sdata[-2] - sdata[-1]))
            try:
                Q_max /= abs((sdata[0] - sdata[-1]))
                Q_maxdiff = (Q_max - q_dict[len(data)], sdata[-1])
            except ZeroDivisionError:
                pass

        if not Q_mindiff[0] > 0 and not Q_maxdiff[0] > 0:
            outliers = []

        elif Q_mindiff[0] == Q_maxdiff[0]:
            outliers = [Q_mindiff[1], Q_maxdiff[1]]

        elif Q_mindiff[0] > Q_maxdiff[0]:
            outliers = Q_mindiff[1]

        else:
            outliers = Q_maxdiff[1]
    else:
        outliers = []

    return outliers


# ==== other useful functions
def compute_stats(values):
    """
    computes mean, standard deviation and standard error of the mean of the input array values
    :param values: array with values
    :return: num_values (double): number of values in the array
             mean       (double): average
             std        (double): standard deviation
             sem        (double): standard error of the mean
    """
    num_values = len(values)
    mean = np.mean(values)
    std = np.std(values)
    sem = std / np.sqrt(num_values)

    return mean, std, sem, num_values


def concatenate_string_info(data, outlier=[]):
    """
    Given a list of table entries and a (potential) index of outliers, this function removes the outliers and then
    generates concatenated strings of information
    :param outlier:
    :param data:
    :param outliers_index:
    :return:
    """
    standard_type = data["standard_type"]
    chembl_id_doc = data["document_chembl_id"]
    chembl_id = data["chembl_id"]

    # in case of outliers, removes the corresponding information
    if outlier:
        standard_type = standard_type.drop(standard_type.index[outlier])
        chembl_id_doc = chembl_id_doc.drop(chembl_id_doc.index[outlier])
        chembl_id = chembl_id.drop(chembl_id.index[outlier])

    # unique values (dirty trick with list and set)
    standard_type = list(set(standard_type))
    chembl_id_doc = list(set(chembl_id_doc))
    chembl_id = list(set(chembl_id))

    # concatenates into a single string (comma separated)
    standard_types = ",".join(standard_type)
    chembl_docs_ids = ",".join(chembl_id_doc)
    chembl_ids = ",".join(chembl_id)

    return chembl_ids, chembl_docs_ids, standard_types



def retrieve_data(target_id='CHEMBL3979', waitbar=True, endpoints = ["Ki", "Kd"], relations = ["="]):
    """
    Function to retrieve the data from ChEMBL, starting from a given ChEMBL ID (target)
    :param target_id (str): ChEMBL target ID
    :param waitbar (bool): display the wait bar. With Jupyter, this might give some display troubles, so set to False
    :return: df (pd dataframe): contains the raw data retrieved from ChEMBL based on the criteria

    """
    activity = new_client.activity
    res = activity.filter(target_chembl_id=target_id)  # only binding assays

    # initialization
    df = pd.DataFrame()

    print("Collecting data from ChEMBL...")

    # checks data and collects them in a dataframe

    try:
        while res:  # runs over the retrieved entries
            entry = res.next()
            # checks whether the compounds satisfy the criteria (endpoints and relation type), if so, stores them
            if entry['standard_type'] in endpoints and entry["relation"] in relations:

                # checks whether some warning flags are present for the activity or the entry
                if entry["data_validity_comment"] is not None:
                    warning_flag = True  # explicit flag in case there are comments on the reliability
                else:
                    warning_flag = False

                # retains only ki/kd and precise values, plus additional information to check for validity and refs
                df = df._append({'smiles': entry["canonical_smiles"], 'standard_type': entry['standard_type'],
                                'value': entry["standard_value"], 'units': entry["standard_units"],
                                'chembl_id': entry["molecule_chembl_id"], 'document_chembl_id': entry["document_chembl_id"],
                                'data_validity_comment': entry["data_validity_comment"],
                                'data_validity_description': entry["data_validity_description"],
                                'activity_comment': entry["activity_comment"], 'warning_flag': warning_flag},
                               ignore_index=True)
    except StopIteration:
        print(str(len(df.index)) + " molecules collected.")

    return df


def curate_struct(table):
    table_curated = table.copy()  # copy for further editing

    print(len(table_curated))

    # structure curation
    for index, row in table_curated.iterrows():
        smiles = row['smiles']
        if smiles is not None:
            smiles, salt, failed_sanit, neutralized = prepare_structures(smiles, remove_salts=True, sanitize=True, neutralize=True)
        else:
            failed_sanit = True
            salt = False
            neutralized = True
            smiles = 'missing'

        table_curated.loc[index, 'IsSalt'] = salt
        table_curated.loc[index, 'FailedSanit'] = failed_sanit
        table_curated.loc[index, 'Neutralized'] = neutralized
        table_curated.loc[index, 'CuratedSmiles'] = smiles

    # "Cleans" the new table (salts & failures in sanitization)
    # removes salts
    table_curated.drop(table_curated.loc[table_curated['IsSalt'] == True].index, inplace=True)
    # removes failed sanitization
    table_curated.drop(table_curated.loc[table_curated['FailedSanit'] == True].index, inplace=True)
    # removes entries with warning flags
    table_curated.drop(table_curated.loc[table_curated['warning_flag'] == True].index, inplace=True)

    return table_curated


def class_vector(table_final, thr_class):
    smiles = pd.DataFrame(table_final['smiles'])
    table_class = pd.DataFrame(smiles)
    table_class['exp_mean [nM]'] = table_final['exp_mean']
    table_class['y'] = 1*(table_final.exp_mean < thr_class)

    return table_class


def prepare_structures(smiles, remove_salts=True, sanitize=True, neutralize=True):
    # initialize the outputs
    salt = []
    failed_sanit = []
    neutralized = []

    # performs the molecule preparation based on the flags
    if remove_salts and smiles is not None:
        if "." in smiles:  # checks if salts
            salt = True
        else:
            salt = False

    if sanitize is True:
        smiles, failed_sanit = sanitize_mol(smiles)

    if neutralize is True and failed_sanit is False:
        smiles, neutralized = neutralize_mol(smiles)

    return smiles, salt, failed_sanit, neutralized


def _InitialiseNeutralisationReactions():
    """ adapted from the rdkit contribution of Hans de Winter """
    patts = (
        # Imidazoles
        ('[n+;H]', 'n'),
        # Amines
        ('[N+;!H0]', 'N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]', 'O'),
        # Thiols
        ('[S-;X1]', 'S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]', 'N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]', 'N'),
        # Tetrazoles
        ('[n-]', '[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]', 'S'),
        # Amides
        ('[$([N-]C=O)]', 'N'),
    )
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]


def sanitize_mol(smiles):
    """ Sanitizes a molecule using rdkit """
    # init
    failed_sanit = False

    # == basic checks on SMILES validity
    mol = Chem.MolFromSmiles(smiles)

    # flags: Kekulize, check valencies, set aromaticity, conjugation and hybridization
    san_opt = Chem.SanitizeFlags.SANITIZE_ALL

    # check if the conversion to mol was successful, return otherwise
    if mol is None:
        failed_sanit = True
    # sanitization based on the flags (san_opt)
    else:
        sanitize_fail = Chem.SanitizeMol(mol, catchErrors=True, sanitizeOps=san_opt)
        if sanitize_fail:
            failed_sanit = True
            raise ValueError(sanitize_fail)  # returns if failed

    return smiles, failed_sanit


# ====== neutralizes charges based on the patterns specified in _InitialiseNeutralisationReactions
def neutralize_mol(smiles):
    neutralized = False
    mol = Chem.MolFromSmiles(smiles)

    # retrieves the transformations
    transfm = _InitialiseNeutralisationReactions()  # set of transformations

    # applies the transformations
    for i, (reactant, product) in enumerate(transfm):
        while mol.HasSubstructMatch(reactant):
            neutralized = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product)
            mol = rms[0]

    # converts back the molecule to smiles
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    return smiles, neutralized


def fetch_chembl(chembl_targetid='CHEMBL2047', bar=True, endpoints=["EC50"], assign_class=True, thr_class=100, rm_high_std=True):

    # fetches the data and saves
    table_raw = retrieve_data(target_id=chembl_targetid, waitbar=bar, endpoints=endpoints)

    # performs structure curation and saves
    print('Curating the structures ...')
    table_curated = curate_struct(table_raw)

    # experimental property curation and saves
    print('Curating the experimental properties ...')
    table_final = curate_exp(table_curated, rm_high_std=rm_high_std)

    # and we're done :)
    print('... Done.')

    n_comp = len(table_final)
    print('Total number of compounds: ' + str(n_comp))

    # assigns a class based on the specified inputs
    if assign_class:
        table_class = class_vector(table_final, thr_class)

        # some printing, just for fun
        n_a = sum(table_class['y'])
        print(' - active compounds: ' + str(n_a))
        print(' - inactive compounds: ' + str(n_comp - n_a))
        print(' - % active compounds: ' + str(round(n_a / n_comp * 100, 2)))

    else:
        table_class = table_final.copy()

    return table_class


def process_new_data(df, dataset_name: str) -> pd.DataFrame:
    activity_threshold = 100  # in nM

    actives = df.loc[df['exp_mean [nM]'] <= activity_threshold, 'smiles'].tolist()
    inactives = df.loc[df['exp_mean [nM]'] > activity_threshold, 'smiles'].tolist()

    # Clean molecules
    actives, actives_failed = clean_mols(actives)
    inactives, inactives_failed = clean_mols(inactives)

    print('Parsing errors:')
    [print(f"{k}: {v}") for k, v in Counter(actives_failed['reason'] + inactives_failed['reason']).items()]

    # Get the sets
    actives_clean = set(actives['clean'])
    inactives_clean = set(inactives['clean'])

    # Check if there are overlapping molecules in both the set of inactives and actives
    intersecting_smiles = actives_clean & inactives_clean
    actives_clean = [smi for smi in actives_clean if smi not in intersecting_smiles]
    inactives_clean = [smi for smi in inactives_clean if smi not in intersecting_smiles]

    # Put it all together
    y = [1] * len(actives_clean) + [0] * len(inactives_clean)
    smiles = list(actives_clean) + list(inactives_clean)

    # put together in a dataframe and shuffle the rows
    df = pd.DataFrame({'smiles': smiles, 'y': y})
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv(f'data/clean/{dataset_name}.csv', index=False)


if __name__ == '__main__':

    # Extra datasets to fetch
    # 'CDK1': 'CHEMBL308',
    # 'MNK1': 'CHEMBL4718'

    #### Download and curate the data ####
    os.chdir(ROOTDIR)

    # Cyclin-dependent kinase 1
    df_CHEMBL308 = fetch_chembl('CHEMBL308', endpoints=['Ki'])
    process_new_data(df_CHEMBL308, 'CHEMBL308_Ki')

    # MAP kinase-interacting serine/threonine-protein kinase MNK1
    df_CHEMBL4718 = fetch_chembl('CHEMBL4718', endpoints=['Ki'])  # this dataset has almost no Ki values (just 54)
    process_new_data(df_CHEMBL4718, 'CHEMBL4718_Ki')
