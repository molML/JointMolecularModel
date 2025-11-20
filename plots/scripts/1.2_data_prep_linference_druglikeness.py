""" Perform model inference for the jcm model

Derek van Tilborg
Eindhoven University of Technology
November 2024
"""

import os
import sys
import warnings
import pandas as pd
from constants import ROOTDIR
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import RDConfig, QED, Descriptors
from rdkit import RDLogger

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

RDLogger.DisableLog('rdApp.*')


if __name__ == '__main__':

    os.chdir(ROOTDIR)

    # read the output file from 1.2_data_prep_inference.R
    df = pd.read_csv('results/screening_libraries/all_inference_data.csv')

    smiles = []
    SA_scores = []
    MW_scores = []
    n_atoms = []
    QED_scores = []

    for smi in tqdm(set(df['smiles'])):
        try:
            mol = Chem.MolFromSmiles(smi)
            sa = sascorer.calculateScore(mol)
            mw = Descriptors.MolWt(mol)
            na = mol.GetNumHeavyAtoms()
            qed = QED.qed(mol)

            smiles.append(smi)
            SA_scores.append(sa)
            MW_scores.append(mw)
            n_atoms.append(na)
            QED_scores.append(qed)
        except:
            pass

    pd.DataFrame({'smiles': smiles,
                  'SA_scores': SA_scores,
                  'MW_scores': MW_scores,
                  'n_atoms': n_atoms,
                  'QED_scores': QED_scores
                  }).to_csv('results/screening_libraries/druglike_descriptors.csv', index=False)
