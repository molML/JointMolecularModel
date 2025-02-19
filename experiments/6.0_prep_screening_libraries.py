"""
Script to clean the data of all screening libraries

- Asinex
- Enamine hit locator
- Enamine screening collection
- Specs

Derek van Tilborg
Februari 2025
Eindhoven University of Technology
"""

import os
import pandas as pd
from cheminformatics.cleaning import clean_mols
from constants import ROOTDIR
from rdkit import Chem

if __name__ == '__main__':

    os.chdir(ROOTDIR)

    libraries = {'asinex': "data/screening_libraries/asinex_03_Feb_2022.sdf",
                 'enamine_hit_locator': "data/screening_libraries/Enamine_Hit_Locator_Library_460160cmpds_20250205.sdf",
                 'enamine_screening_collection': "data/screening_libraries/Enamine_screening_collection_202412.sdf",
                 'specs': "data/screening_libraries/Specs_ExAcD_Aug_2020.sdf"}

    for library_name, library_path in libraries.items():

        print(library_name)

        # Read sdf
        library_smiles = [Chem.MolToSmiles(mol) for mol in Chem.SDMolSupplier("data/screening_libraries/asinex_03_Feb_2022.sdf") if mol is not None]

        # Clean SMILES
        clean_smiles, failed_smiles = clean_mols(library_smiles)

        # Save file
        df = pd.DataFrame({'smiles_original': clean_smiles['original'], 'smiles_cleaned': clean_smiles['clean']})
        df.to_csv(f"data/screening_libraries/{library_name}_cleaned.csv")
