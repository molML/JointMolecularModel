""" Filter the screening libraries for prospective screening. We flag all molecules that don't meet certain physchem
rules that possibly prent the molecule from dissolving and we flag molecules with some highly reactive groups that
might be problematic in our assay

Derek van Tilborg
Eindhoven University of Technology
April 2025
"""

import os
import pandas as pd
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from cheminformatics.cleaning import clean_single_mol
from constants import ROOTDIR


def find_physchem_violations(smiles: str) -> bool:
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
        return "SMILES issues"

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

    if rot_bonds > 12:
        found_reasons.append('Rotatable bonds')  # Highly flexible, might cause solubility issues

    rule_of_five_violations = 1 * (mw > 500) + 1 * (logp > 5) + 1 * (hbd > 5) + 1 * (hba > 10)
    if rule_of_five_violations > 2:
        found_reasons.append('Ro5 violation')  # Too many issues

    if heavy_atoms < 12:
        found_reasons.append('Too small')   # Too small to reasonably be a kinase inhibitor

    if rings == 0:
        found_reasons.append('No rings')   # not enough rings, every kinase inhibitor yet has 1+

    # Passes all filters
    return ", ".join(found_reasons) if found_reasons else 'Passed'


def find_reactivity_violations(smiles: str) -> str:
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
        return "SMILES issues"

    found_groups = []
    for name, pattern in reactivity_patterns.items():
        if mol.HasSubstructMatch(pattern):
            found_groups.append(name)

    return ", ".join(found_groups) if found_groups else 'Passed'


def process_specs_sdf(sdf_path):

    # Read sdf and clean SMILES
    supplier = Chem.SDMolSupplier(sdf_path)

    library_data = []
    for mol in tqdm(supplier):
        try:
            smi = Chem.MolToSmiles(mol)
            smi_clean, cleaning_violations = clean_single_mol(smi)

            mol_id = mol.GetProp("IDNUMBER")
            mg_available = float(mol.GetProp('AVAILABLE'))
            url = mol.GetProp('ADD_INFO')
            reactivity_violations = find_reactivity_violations(smi)
            physchem_violations = find_physchem_violations(smi)
            library_data.append((mol_id, smi, smi_clean, cleaning_violations, reactivity_violations, physchem_violations, mg_available, url))
        except:
            continue

    library_df = pd.DataFrame(library_data,
                              columns=["specs_ID", "smiles_original", "smiles_cleaned", "cleaning_violations",
                                       "reactivity_violations", "physchem_violations", "mg_available", 'url'])

    return library_df


if __name__ == '__main__':

    os.chdir(ROOTDIR)

    datasets = ['CHEMBL4718_Ki', 'CHEMBL308_Ki', 'CHEMBL2147_Ki']
    libraries = {'specs_10': "data/screening_libraries/specs_2025/Specs_SC_10mg_Apr2025.sdf",
                 'specs_20': "data/screening_libraries/specs_2025/Specs_SC_20mg_Apr2025.sdf",
                 'specs_50': "data/screening_libraries/specs_2025/Specs_SC_50mg_Apr2025.sdf"}

    # Process all Specs libraries and put them together
    specs = []
    for library_name, library_path in libraries.items():
        print(library_name)
        specs.append(process_specs_sdf(library_path))
    specs = pd.concat(specs)

    specs.to_csv(f"data/screening_libraries/specs_2025/specs_raw_Apr2025.csv", index=False)

    # remove molecules that didn't pass the cleaning step
    specs = specs[specs['cleaning_violations'] == 'Passed']

    # remove molecules that didn't pass the reactivity filters
    specs = specs[specs['reactivity_violations'] == 'Passed']

    # remove molecules that didn't pass the physchem filter
    specs = specs[specs['physchem_violations'] == 'Passed']

    # remove molecules that are not available to order. We take 10mg to be safe (we realistically only need 100 µg)
    specs = specs[specs['mg_available'] >= 10]

    # remove duplicates
    specs = specs.drop_duplicates(subset="specs_ID")

    # Save file
    specs.to_csv(f"data/screening_libraries/specs_2025/specs_clean_Apr2025.csv", index=False)
