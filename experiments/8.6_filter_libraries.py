""" Filter the screening libraries for prospective screening. We flag all molecules that don't meet certain physchem
rules that possibly prent the molecule from dissolving and we flag molecules with some highly reactive groups that
might be problematic in our assay

Derek van Tilborg
Eindhoven University of Technology
April 2025
"""


import os
from collections import Counter
import pandas as pd
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
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
    # rings = Descriptors.RingCount(mol)  # Number of rings

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

    # Passes all filters
    return ", ".join(found_reasons) if found_reasons else "None"


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

    return ", ".join(found_groups) if found_groups else "None"


if __name__ == '__main__':

    os.chdir(ROOTDIR)

    data_dir = os.path.join('results', 'prospective')

    datasets = ['CHEMBL4718_Ki', 'CHEMBL308_Ki', 'CHEMBL2147_Ki']
    library_names = ['specs', 'asinex', 'enamine_hit_locator']

    all_screening_results = []
    for dataset in datasets:
        for method in ['smiles_jmm']:
            print(f"{dataset} - {method}")

            df = pd.read_csv(os.path.join(data_dir, method, dataset, 'results_preds.csv'))
            df = df[df['split'].isin(library_names)]
            df['method'] = method

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
    all_screening_results_df.to_csv(os.path.join(data_dir, 'all_screening_results.csv'), index=False)
