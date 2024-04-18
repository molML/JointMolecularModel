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


VOCAB = {'start_char': ':', 'end_char': ';', 'pad_char': '_', 'max_len': 62, 'vocab_size': 35,
         'start_idx': 0, 'end_idx': 33, 'pad_idx': 34,
         'indices_token': {0: ':', 1: 'C', 2: 'c', 3: '(', 4: ')', 5: 'O', 6: '1', 7: '=', 8: 'N', 9: '2', 10: '@',
                           11: '[', 12: ']', 13: 'H', 14: 'n', 15: '3', 16: 'F', 17: '4', 18: 'S', 19: '/', 20: 'Cl',
                           21: 's', 22: '5', 23: 'o', 24: '#', 25: '\\', 26: 'Br', 27: 'P', 28: '6', 29: 'I', 30: '7',
                           31: '8', 32: 'p', 33: ';', 34: '_'},
         'token_indices': {':': 0, 'C': 1, 'c': 2, '(': 3, ')': 4, 'O': 5, '1': 6, '=': 7, 'N': 8, '2': 9, '@': 10,
                           '[': 11, ']': 12, 'H': 13, 'n': 14, '3': 15, 'F': 16, '4': 17, 'S': 18, '/': 19, 'Cl': 20,
                           's': 21, '5': 22, 'o': 23, '#': 24, '\\': 25, 'Br': 26, 'P': 27, '6': 28, 'I': 29, '7': 30,
                           '8': 31, 'p': 32, ';': 33, '_': 34}}
