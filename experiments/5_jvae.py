
import os
import itertools
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from jcm.junkyard import EcfpJVAE
from jcm.trainer import train_jvae
from jcm.datasets import load_moleculeace
from jcm.config import Config
from constants import ROOTDIR
from jcm.utils import to_binary, ClassificationMetrics, logits_to_pred, reconstruction_metrics
from eval.ood import pred, ood_correlations, distance_to_trainset, evaluate_predictions
import pandas as pd
import matplotlib.pyplot as plt

MLP_DIR = os.path.join(ROOTDIR, 'results/pre_trained_mlps')
VAE_PATH = os.path.join(ROOTDIR, 'results/chembl_vae/pretrained_vae_100000.pt')
VAL_SPLIT = 0.2
MAX_ITERS = 100
BATCH_SIZE = 128
MLP_LOSS_SCALAR = 1
FREEZE_VAE = False
FOLDS = 5

jvae_hypers = {'input_dim': 2048, 'latent_dim': 64, 'hidden_dim_vae': 2048, 'out_dim_vae': 2048,
               'beta': 0.001, 'n_layers_mlp': 2, 'hidden_dim_mlp': 2048, 'anchored': True,
               'l2_lambda': 1e-4, 'n_ensemble': 10, 'output_dim_mlp': 2, 'class_scaling_factor': 40,
               'variational_scale': 0.1, 'device': None, 'mlp_loss_scalar': 1}


# [x] TODO pretrain EcfpVAE
# [x] TODO pretrain MLPs
# [x] TODO merge EcfpVAE + MLPs
# [ ] TODO reconstruction loss
# [ ] TODO Train EcfpJVAE
# [ ] TODO OOD quantification


if __name__ == '__main__':

    datasets = [f'data/moleculeace/{i}' for i in os.listdir('data/moleculeace') if i.startswith('CHEMBL')]
    all_results = defaultdict(list)

    for dataset in tqdm(datasets):
        dataset_name = dataset.split('/')[-1].split('.csv')[0]

        for fold in range(FOLDS):

            train_dataset, val_dataset, test_dataset = load_moleculeace(dataset, val_split=VAL_SPLIT, seed=fold)

            config = Config(max_iters=0, batch_size=BATCH_SIZE, batch_end_callback_every=20, out_path=None)
            jvae_hypers['mlp_loss_scalar'] = MLP_LOSS_SCALAR
            config.set_hyperparameters(**jvae_hypers)

            model = EcfpJVAE(**config.hyperparameters)
            pretrained_ensemble = torch.load(os.path.join(MLP_DIR, f"{dataset_name}_mlp.pt")).ensemble

            for _ in range((MAX_ITERS//20)+1):

                model, trainer = train_jvae(config, train_dataset, val_dataset,
                                            freeze_vae=False,
                                            freeze_mlp=False,
                                            pre_trained_path_mlp=pretrained_ensemble,
                                            pre_trained_path_vae=VAE_PATH)

                results = {'dataset': dataset_name,
                           'fold': fold,
                           'iters': config.max_iters,
                           'freeze_vae': FREEZE_VAE,
                           'mlp_loss_scalar': MLP_LOSS_SCALAR}

                # Eval
                train_results = evaluate_predictions(model, config, train_dataset, train_dataset)
                val_results = evaluate_predictions(model, config, val_dataset, train_dataset)
                test_results = evaluate_predictions(model, config, test_dataset, train_dataset)

                train_results = {'train_' + k: v for k, v in train_results.items()}
                val_results = {'val_' + k: v for k, v in val_results.items()}
                test_results = {'test_' + k: v for k, v in test_results.items()}

                results = results | train_results | val_results | test_results

                for k, v in results.items():
                    all_results[k].append(v)

                pd.DataFrame(all_results).to_csv('jvae_results.csv')
                config.max_iters += 20



