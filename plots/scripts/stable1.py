"""
Script lookup data set sizes for supplementary table 1

Derek van Tilborg
January 2025
Eindhoven University of Technology
"""

import os
import pandas as pd
from constants import ROOTDIR

if __name__ == '__main__':

    os.chdir(ROOTDIR)

    # Define the data
    dataset_names = pd.DataFrame({
        "id": [
            "PPARG", "Ames_mutagenicity", "ESR1_ant", "TP53", "CHEMBL1871_Ki", "CHEMBL218_EC50",
            "CHEMBL244_Ki", "CHEMBL236_Ki", "CHEMBL234_Ki", "CHEMBL219_Ki", "CHEMBL238_Ki", "CHEMBL4203_Ki",
            "CHEMBL2047_EC50", "CHEMBL4616_EC50", "CHEMBL2034_Ki", "CHEMBL262_Ki", "CHEMBL231_Ki", "CHEMBL264_Ki",
            "CHEMBL2835_Ki", "CHEMBL2971_Ki", "CHEMBL237_EC50", "CHEMBL237_Ki", "CHEMBL233_Ki", "CHEMBL4792_Ki",
            "CHEMBL239_EC50", "CHEMBL3979_EC50", "CHEMBL235_EC50", "CHEMBL4005_Ki", "CHEMBL2147_Ki", "CHEMBL214_Ki",
            "CHEMBL228_Ki", "CHEMBL287_Ki", "CHEMBL204_Ki", "CHEMBL1862_Ki"
        ],
        "name": [
            "PPARyl", "Ames", "ESR1", "TP53", "AR", "CB1", "FX", "DOR", "D3R", "D4R", "DAT", "CLK4",
            "FXR", "GHSR", "GR", "GSK3", "HRH1", "HRH3", "JAK1", "JAK2", "KOR (a)", "KOR (i)", "MOR", "OX2R",
            "PPARa", "PPARym", "PPARd", "PIK3CA", "PIM1", "5-HT1A", "SERT", "SOR", "Thrombin", "ABL1"
        ]
    })

    table_S1 = {'Dataset': [],
                'Pharmalogical target': [],
                'Endpoint': [],
                'Original size': [],
                'Curated size': [],
                'Train size': [],
                'TestID size': [],
                'TestOOD size': []}


    # ChEMBL
    table_S1['Dataset'].append('ChEMBL v33')
    table_S1['Pharmalogical target'].append('-')
    table_S1['Endpoint'].append('-')
    table_S1['Original size'].append(len(pd.read_table("data/ChEMBL/chembl_33_chemreps.txt")))
    table_S1['Curated size'].append(len(pd.read_csv('data/clean/ChEMBL_33_filtered.csv')))
    chembl_df = pd.read_csv('data/split/ChEMBL_33_split.csv')
    table_S1['Train size'].append(len(chembl_df[chembl_df['split'] == 'train']))
    table_S1['TestID size'].append(len(chembl_df[chembl_df['split'] == 'test']))
    table_S1['TestOOD size'].append('-')


    # MoleculeACE
    moleculeace_datasets = [i for i in os.listdir('data/clean') if i.startswith('CHEMBL')]
    for moleculeace_name in moleculeace_datasets:
        table_S1['Dataset'].append(moleculeace_name.split('_')[0])
        table_S1['Pharmalogical target'].append(dataset_names[dataset_names['id'] == moleculeace_name.split('.')[0]]['name'].item())
        table_S1['Endpoint'].append(f"Bioactivity ({moleculeace_name.split('_')[1].split('.')[0]})")
        table_S1['Original size'].append(len(pd.read_csv(f'data/moleculeace_original/{moleculeace_name}')))
        table_S1['Curated size'].append(len(pd.read_csv(f'data/clean/{moleculeace_name}')))
        moleculace_df = pd.read_csv(f"data/split/{moleculeace_name.split('.')[0]}_split.csv")
        table_S1['Train size'].append(len(moleculace_df[moleculace_df['split'] == 'train']))
        table_S1['TestID size'].append(len(moleculace_df[moleculace_df['split'] == 'test']))
        table_S1['TestOOD size'].append(len(moleculace_df[moleculace_df['split'] == 'ood']))


    # LitPCBA
    for litpcba_name in ['ESR1_ant', 'TP53', 'PPARG']:

        table_S1['Dataset'].append(f'{litpcba_name}_LitPCBA')
        table_S1['Pharmalogical target'].append(f'{litpcba_name}')
        table_S1['Endpoint'].append('Bioactivity')
        litpcba_mols = []
        with open(f'data/LitPCBA/{litpcba_name}/actives.smi', 'r') as file:
            litpcba_mols = [line.strip().split(' ')[0] for line in file]
        with open(f'data/LitPCBA/{litpcba_name}/inactives.smi', 'r') as file:
            litpcba_mols = litpcba_mols + [line.strip().split(' ')[0] for line in file]
        table_S1['Original size'].append(len(litpcba_mols))
        table_S1['Curated size'].append(len(pd.read_csv(f'data/clean/{litpcba_name}.csv')))
        litpcba_df = pd.read_csv(f'data/split/{litpcba_name}_split.csv')
        table_S1['Train size'].append(len(litpcba_df[litpcba_df['split'] == 'train']))
        table_S1['TestID size'].append(len(litpcba_df[litpcba_df['split'] == 'test']))
        table_S1['TestOOD size'].append(len(litpcba_df[litpcba_df['split'] == 'ood']))


    # Ames
    ames_mols = []
    with open('data/Ames_mutagenicity/smiles_cas_N6512.smi', 'r') as file:
        for line in file:
            label = int(line.strip().split('\t')[-1])
            ames_mols.append(line.strip().split(' ')[0])

    table_S1['Dataset'].append('Ames mutagenicity')
    table_S1['Pharmalogical target'].append('-')
    table_S1['Endpoint'].append('Mutagenicity')
    table_S1['Original size'].append(len(ames_mols))
    table_S1['Curated size'].append(len(pd.read_csv('data/clean/Ames_mutagenicity.csv')))
    ames_df = pd.read_csv('data/split/Ames_mutagenicity_split.csv')
    table_S1['Train size'].append(len(ames_df[ames_df['split'] == 'train']))
    table_S1['TestID size'].append(len(ames_df[ames_df['split'] == 'test']))
    table_S1['TestOOD size'].append(len(ames_df[ames_df['split'] == 'ood']))


    # write to file
    df = pd.DataFrame(table_S1)
    df.to_csv('plots/tables/s_table_2.csv', index=False)
