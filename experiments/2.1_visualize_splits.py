"""
Plot the finetuning dataset splits using a TSNE projection

Derek van Tilborg
Eindhoven University of Technology
June 2024
"""

import os
from os.path import join as ospj
import pandas as pd
from rdkit import Chem
from cheminformatics.descriptors import mols_to_ecfp
from cheminformatics.utils import get_scaffold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from constants import ROOTDIR


def tsne_mols(mols: list, split: list[str], **kwargs) -> pd.DataFrame:
    """ Perform a TSNE on a set of molcules using their ECFPs """
    all_ecfps = mols_to_ecfp(mols, radius=2, nbits=2048, to_array=True)

    # TSNE
    reducer = TSNE(n_components=2, **kwargs)
    projection = reducer.fit_transform(all_ecfps)

    # Create a DataFrame for the UMAP results
    df = pd.DataFrame(projection, columns=['x', 'y'])

    df['Split'] = pd.Categorical(split, categories=['ood', 'train', 'test'], ordered=True)

    return df


if __name__ == '__main__':

    IN_DIR_PATH = 'data/split'
    OUT_DIR_PATH = 'results/dataset_clustering'
    os.chdir(ROOTDIR)

    datasets = [i for i in os.listdir(IN_DIR_PATH) if i.endswith('split.csv')]
    datasets = [i for i in datasets if i != 'ChEMBL_33_split.csv']

    for dataset in tqdm(datasets):

        df = pd.read_csv(ospj(IN_DIR_PATH, dataset))
        smiles = df['smiles'].tolist()

        # Get molecules and their scaffolds
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        scaffold_mols = [get_scaffold(m, scaffold_type='cyclic_skeleton') for m in mols]

        # perform TSNE
        projection_mols = tsne_mols(mols, split=df['split'], perplexity=30)
        projection_scaffolds = tsne_mols(scaffold_mols, split=df['split'], perplexity=30)

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        # Create the first scatter plot
        sns.scatterplot(ax=axes[0], x='x', y='y', hue='Split', data=projection_mols,
                        palette=['#e9c46a', '#0a9396', '#94d2bd'])
        axes[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axes[0].set_xlabel('Whole molecule ECFPs')
        axes[0].set_ylabel('')

        # Create the second scatter plot
        sns.scatterplot(ax=axes[1], x='x', y='y', hue='Split', data=projection_scaffolds,
                        palette=['#e9c46a', '#0a9396', '#94d2bd'])
        axes[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axes[1].set_xlabel('Cyclic skeleton ECFPs')
        axes[1].set_ylabel('')

        fig.suptitle(f'TSNE of {dataset.replace("_split.csv", "")} (n={len(smiles)})')
        plt.savefig(ospj(OUT_DIR_PATH, dataset.replace('.csv', '.pdf')))
        plt.show()
