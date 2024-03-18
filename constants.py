import os
import torch

ROOTDIR = "/Users/derekvantilborg/Dropbox/PycharmProjects/JointChemicalModel"


DEFAULT_CONFIG = {'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                  'batch_size': 128,
                  'num_workers': 1,
                  'grad_norm_clip': 1,
                  'batch_end_callback_every': 1000,
                  'val_molecules_to_sample': 1000,
                  'out_path': os.path.join(ROOTDIR, 'results/chembl_vae')}


VAE_PRETRAIN_HYPERPARAMETERS = {'input_dim': 2048,
                                "latent_dim": 64,
                                'out_dim': 2048,
                                'beta': 0.001,
                                'class_scaling_factor': 40,
                                'variational_scale': 0.1,
                                'lr': 3e-4}

