""" Filter the screening libraries for prospective screening. We flag all molecules that don't meet certain physchem
rules that possibly prent the molecule from dissolving and we flag molecules with some highly reactive groups that
might be problematic in our assay

Derek van Tilborg
Eindhoven University of Technology
April 2025
"""

import os
from collections import Counter
from itertools import batched
from os.path import join as ospj
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from cheminformatics.multiprocessing import tanimoto_matrix
from cheminformatics.utils import smiles_to_mols, get_scaffold
from rdkit.Chem import rdFingerprintGenerator
from constants import ROOTDIR


def physchem_violations(smiles: str) -> bool:
    """ Checks if a molecule is likely soluble in DMSO based on some pysicochemical properties. We use slightly relaxed
        RoF and Veber rules
        1. MW should be 200-600
        2. LogP should be <= 6
        3. TPSA should be 20–140 Å²
        4. No. HBD should be < 6
        5. No. rot. bonds should be <= 10
        6. Rule of five violations should be <= 2

    :param smiles: SMILES string
    :return:  True if likely soluble, False otherwise
    """

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "Invalid SMILES"

    # Compute molecular properties
    mw = Descriptors.MolWt(mol)  # Molecular weight
    logp = Crippen.MolLogP(mol)  # LogP (lipophilicity)
    tpsa = Descriptors.TPSA(mol)  # Topological Polar Surface Area
    hbd = Descriptors.NumHDonors(mol)  # Number of hydrogen bond donors
    hba = Descriptors.NumHAcceptors(mol)  # Number of hydrogen bond acceptors
    rot_bonds = Descriptors.NumRotatableBonds(mol)  # Rotatable bonds
    rings = Descriptors.RingCount(mol)  # Number of rings
    heavy_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)

    found_reasons = []

    if not 200 <= mw <= 600:
        found_reasons.append('MW')  # Too large or too small

    if logp > 6:  # although there are some kinases with a LogP of up to 7
        found_reasons.append('LogP')  # Too hydrophobic

    if not 20 <= tpsa <= 150:
        found_reasons.append('TPSA')  # Too polar, may precipitate or not polar enough

    if hbd > 5:
        found_reasons.append('HBD')  # Too many hydrogen bond donors

    # if rings > 4 and logp > 4:
    #     return False  # High aromaticity + lipophilicity → aggregation risk

    if rot_bonds > 12:
        found_reasons.append('Rotatable bonds')  # Highly flexible, might cause solubility issues

    rule_of_five_violations = 1 * (mw > 500) + 1 * (logp > 5) + 1 * (hbd > 5) + 1 * (hba > 10)
    if rule_of_five_violations > 2:
        found_reasons.append('Ro5 violation')  # Too many issues

    if heavy_atoms < 12:
        found_reasons.append('Too small')   # Too small to reasonably be a kinase inhibitor

    if rings < 2:
        found_reasons.append('Less than two rings')   # not enough rings, every kinase inhibitor yet has 2+

    # Passes all filters
    return ", ".join(found_reasons) if found_reasons else None


def reactivity_violations(smiles: str) -> str:
    """Check if a molecule contains any highly reactive functional groups that could degrade or interfere with kinase assays reactive groups."""

    # Define SMARTS patterns for undesired groups
    reactivity_patterns = {
        "Terminal_Enone": Chem.MolFromSmarts("C=CC(=O)[C,N]"),  # terminal Michael acceptors (might be covalent warheads)
        "Isocyanate": Chem.MolFromSmarts("N=C=O"),  # Highly reactive with nucleophiles (e.g., lysines, buffer amines)
        "Quinone": Chem.MolFromSmarts("O=C1C=CC=C(C1=O)"),  # Redox cycling, cytotoxic, interferes with readouts
        "Nitro_Aromatic": Chem.MolFromSmarts("[NX3](=O)=O"),   # Toxic, redox-active
        "Azide": Chem.MolFromSmarts("N=[N+]=[N-]"),  # Explosive, can form reactive intermediates
        "Epoxide": Chem.MolFromSmarts("C1OC1")  # Strong electrophiles, reactive with thiols and amines
    }

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "Invalid SMILES"

    found_groups = []
    for name, pattern in reactivity_patterns.items():
        if mol.HasSubstructMatch(pattern):
            found_groups.append(name)

    return ", ".join(found_groups) if found_groups else None


def get_all_mol_info(all_library_smiles, all_dataset_smiles):

    all_mol_info_path = ospj(data_dir, 'all_mol_info.pt')

    if os.path.exists(all_mol_info_path):
        all_mol_info = torch.load(all_mol_info_path)
    else:
        all_smiles = list(set(all_dataset_smiles + all_library_smiles))

        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

        all_mol_info = {}
        for smi in tqdm(all_smiles):
            try:
                mol = smiles_to_mols(smi)
                scaffold_mol = get_scaffold(mol, scaffold_type='cyclic_skeleton')
                ecfp = mfpgen.GetFingerprint(mol)
                ecfp_scaffold = mfpgen.GetFingerprint(scaffold_mol)
                all_mol_info[smi] = {'mol': mol, 'ecfp': ecfp, 'ecfp_scaffold': ecfp_scaffold}
            except:
                print(f"failed {smi}")

        torch.save(all_mol_info, all_mol_info_path)

    return all_mol_info


def compute_dataset_library_distance(all_library_smiles, dataset_name, all_mol_info):

    # get the SMILES from the pre-processed file
    data_path = ospj(f'data/clean/{dataset_name}.csv')
    df = pd.read_csv(data_path)
    dataset_smiles = df.smiles.tolist()

    # if there are any failed smiles, remove them
    dataset_smiles = [smi for smi in dataset_smiles if smi in all_mol_info]
    all_library_smiles = [smi for smi in all_library_smiles if smi in all_mol_info]

    # chunk all smiles in batches for efficient multi-threading with manageable memory
    all_library_smiles_batches = [i for i in batched(all_library_smiles, 10000)]

    # tanimoto sim between every screening mol and the training mols
    train_ecfps = [all_mol_info[smi]['ecfp'] for smi in dataset_smiles]
    ECFPs_S_mean = []
    ECFPs_S_max = []
    for batch in tqdm(all_library_smiles_batches, desc='\tComputing Tanimoto similarity', unit_scale=10000):
        T_to_train = tanimoto_matrix([all_mol_info[smi]['ecfp'] for smi in batch],
                                     train_ecfps, take_mean=False)

        ECFPs_S_mean.append(np.mean(T_to_train, 1))
        ECFPs_S_max.append(np.max(T_to_train, 1))
    ECFPs_S_mean = np.concatenate(ECFPs_S_mean)
    ECFPs_S_max = np.concatenate(ECFPs_S_max)

    # tanimoto scaffold sim between every screening mol and the training mols
    train_scaffold_ecfps = [all_mol_info[smi]['ecfp_scaffold'] for smi in dataset_smiles]
    ECFPs_scaff_S_mean = []
    ECFPs_scaff_S_max = []
    for batch in tqdm(all_library_smiles_batches, desc='\tComputing Tanimoto scaffold similarity', unit_scale=10000):
        T_to_train = tanimoto_matrix([all_mol_info[smi]['ecfp_scaffold'] for smi in batch],
                                     train_scaffold_ecfps, take_mean=False)
        ECFPs_scaff_S_mean.append(np.mean(T_to_train, 1))
        ECFPs_scaff_S_max.append(np.max(T_to_train, 1))
    ECFPs_scaff_S_mean = np.concatenate(ECFPs_scaff_S_mean)
    ECFPs_scaff_S_max = np.concatenate(ECFPs_scaff_S_max)

    df = {'smiles': all_library_smiles,
          'dataset': dataset_name,
          'Tanimoto_to_train_mean': ECFPs_S_mean,
          'Tanimoto_to_train_max': ECFPs_S_max,
          'Tanimoto_scaffold_to_train_mean': ECFPs_scaff_S_mean,
          'Tanimoto_scaffold_to_train_max': ECFPs_scaff_S_max
          }

    return pd.DataFrame(df)


if __name__ == '__main__':

    os.chdir(ROOTDIR)

    data_dir = ospj('results', 'prospective')

    datasets = ['CHEMBL4718_Ki', 'CHEMBL308_Ki', 'CHEMBL2147_Ki']
    library_names = ['specs', 'asinex', 'enamine_hit_locator']

    all_screening_results = []
    for dataset in datasets:
        for method in ['smiles_jmm']:
            print(f"{dataset} - {method}")

            df = pd.read_csv(ospj(data_dir, method, dataset, 'results_preds.csv'))
            df = df[df['split'].isin(library_names)]
            df['method'] = method
            df['dataset'] = dataset

            # df.columns
            agg_funcs = {
                col: 'mean' if pd.api.types.is_numeric_dtype(df[col]) else (
                    lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
                for col in df.columns if col not in ['smiles', 'split']
            }

            # Group by two columns and aggregate
            grouped_df = df.groupby(['smiles', 'split']).agg(agg_funcs).reset_index()
            grouped_df = grouped_df.drop('seed', axis=1)

            print(f"\t > library size: {dict(Counter(grouped_df['split']))}")

            grouped_df['physchem_violations'] = [physchem_violations(smi) for smi in tqdm(grouped_df['smiles'])]
            grouped_df['reactivity_violations'] = [reactivity_violations(smi) for smi in tqdm(grouped_df['smiles'])]

            all_screening_results.append(grouped_df)
    all_screening_results_df = pd.concat(all_screening_results, ignore_index=True)

    # Remove all molecules that didn't pass the filters
    all_screening_results_df = all_screening_results_df[all_screening_results_df['reactivity_violations'].isna()]
    all_screening_results_df = all_screening_results_df[all_screening_results_df['physchem_violations'].isna()]

    # Get rid of all columns that are not needed later on
    all_screening_results_df = all_screening_results_df[['smiles', 'split', 'reconstruction_loss', 'edit_distance',
                                                         'y_hat', 'y_unc', 'y_E', 'method', 'dataset']]
    all_screening_results_df = all_screening_results_df.rename(columns={'reconstruction_loss': 'unfamiliarity'})

    # precompute all fingerprints and stuff
    all_dataset_smiles = list(set(sum([pd.read_csv(ospj(f'data/clean/{ds}.csv')).smiles.tolist() for ds in datasets], [])))
    all_library_smiles = list(set(all_screening_results_df.smiles))
    all_mol_info = get_all_mol_info(all_library_smiles, all_dataset_smiles)

    # calculate distances to the train data
    dataset_distances = []
    for dataset_name in datasets:
        dataset_distances.append(compute_dataset_library_distance(all_library_smiles, dataset_name, all_mol_info))
    dataset_distances = pd.concat(dataset_distances)

    # combine the distances with the inference results
    all_screening_results_df = all_screening_results_df.merge(
        dataset_distances,
        how="left",
        on=["smiles", "dataset"]
    )

    # write to csv
    all_screening_results_df.to_csv(ospj(data_dir, 'all_screening_results.csv'), index=False)
