""" Compute a bunch of metrics on the split data

Derek van Tilborg
Eindhoven University of Technology
Augustus 2024
"""

import os
from os.path import join as ospj
from tqdm.auto import tqdm
import pandas as pd
from jcm.training_logistics import get_all_dataset_names, load_dataset_df
from cheminformatics.molecular_similarity import tani_sim_to_train, mean_cosine_cats_to_train, \
    substructure_sim_to_train, applicability_domain_kNN, applicability_domain_SDC
from cheminformatics.complexity import molecular_complexity
from constants import ROOTDIR


if __name__ == '__main__':

    # move to root dir
    os.chdir(ROOTDIR)

    all_dataset_names = get_all_dataset_names()

    for dataset_name in tqdm(all_dataset_names):
        print(dataset_name)

        # load the data and get rid of all columns that might be in there
        df = load_dataset_df(dataset_name)
        df = df.loc[:, ['smiles', 'y', 'cluster', 'split']]

        # get the smiles strings
        train_smiles = df[df['split'] == 'train'].smiles.tolist()
        all_smiles = df.smiles.tolist()

        # ECFP Tanimoto similarity
        tani = tani_sim_to_train(all_smiles, train_smiles, radius=2, nbits=2048, scaffold=False)
        df['Tanimoto_to_train'] = tani

        tani_scaf = tani_sim_to_train(all_smiles, train_smiles, radius=2, nbits=2048, scaffold=True)
        df['Tanimoto_scaffold_to_train'] = tani_scaf

        # Cats Cosine similarity
        cats_cos = mean_cosine_cats_to_train(all_smiles, train_smiles)
        df['Cats_cos'] = cats_cos

        # fractional MSC
        frMSC = substructure_sim_to_train(all_smiles, train_smiles, scaffold=False)
        df['frMSC'] = frMSC

        # Compute "complexity" of molecules
        complex = [molecular_complexity(smi) for smi in tqdm(all_smiles)]
        df = pd.concat((df, pd.DataFrame(complex)), axis=1)

        # KNN OOD metric; 10.1021/acs.chemrestox.9b00498
        knn_ad = applicability_domain_kNN(all_smiles, train_smiles, k=10, sim_cutoff=0.25)
        df['knn_ad'] = knn_ad

        # Sum of distance-weighted contributions; 10.1021/acs.jcim.8b00597
        sdc_ad = applicability_domain_SDC(all_smiles, train_smiles)
        df['sdc_ad'] = sdc_ad

        # write to file
        df.to_csv(ospj('data', 'datasets_with_metrics', f'{dataset_name}.csv'), index=False)
