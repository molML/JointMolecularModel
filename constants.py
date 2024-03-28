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


VOCAB = {'start_char': ':', 'end_char': ';', 'pad_char': '_', 'max_len': 100, 'vocab_size': 39,
         'start_idx': 0, 'end_idx': 37, 'pad_idx': 38,
         'indices_token': {0: ':', 1: 'C', 2: '(', 3: ')', 4: 'N', 5: '1', 6: '=', 7: 'c', 8: '2', 9: 's', 10: 'n',
                           11: '/', 12: 'O', 13: 'I', 14: 'o', 15: 'Cl', 16: '-', 17: '3', 18: '\\', 19: 'S', 20: '[',
                           21: 'H', 22: ']', 23: '4', 24: '+', 25: 'Br', 26: 'F', 27: 'P', 28: '#', 29: '5', 30: 'Se',
                           31: '6', 32: 'Si', 33: 'B', 34: '7', 35: '8', 36: 'se', 37: 'p', 38: ';', 39: '_'},
         'token_indices': {':': 0, 'C': 1, '(': 2, ')': 3, 'N': 4, '1': 5, '=': 6, 'c': 7, '2': 8, 's': 9, 'n': 10,
                           '/': 11, 'O': 12, 'I': 13, 'o': 14, 'Cl': 15, '-': 16, '3': 17, '\\': 18, 'S': 19, '[': 20,
                           'H': 21, ']': 22, '4': 23, '+': 24, 'Br': 25, 'F': 26, 'P': 27, '#': 28, '5': 29, 'Se': 30,
                           '6': 31, 'Si': 32, 'B': 33, '7': 34, '8': 35, 'se': 36, 'p': 37, ';': 38, '_': 39}}
