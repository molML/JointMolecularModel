
import torch
import pandas as pd
from jcm.utils import to_binary, ClassificationMetrics
from dataprep.utils import smiles_to_mols
from dataprep.descriptors import mols_to_ecfp, mols_to_maccs
from jcm.datasets import MoleculeDataset
from jcm.model import VAE
from jcm.trainer import Trainer
from jcm.config import Config
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler
import os
from tqdm import tqdm


if __name__ == '__main__':

    train_smiles = pd.read_csv('data/ChEMBL/train_smiles.csv').smiles.tolist()
    train_dataset = MoleculeDataset(train_smiles, descriptor='ecfp')

    val_smiles = pd.read_csv('data/ChEMBL/val_smiles.csv').smiles.tolist()
    val_dataset = MoleculeDataset(val_smiles, descriptor='ecfp')

    hyperparameters = {'input_dim': 1024,
                       "latent_dim": 128,
                       'out_dim': 1024,
                       'beta': 0.001,
                       'class_scaling_factor': 20,
                       'lr': 3e-4}

    config = Config(max_iters=50000, batch_size=128)
    config.set_hyperparameters(**hyperparameters)

    model = VAE(**config.hyperparameters)
    T = Trainer(config, model, train_dataset)

    def batch_end_callback(trainer):
        if trainer.iter_num % 2000 == 0 and trainer.iter_num > 0:
            balanced_accuracies = []
            losses = []

            ckpt_path = os.path.join(T.config.out_path, f"pretrained_vae_{trainer.iter_num}.pt")
            torch.save(model.state_dict(), ckpt_path)

            val_loader = DataLoader(val_dataset,
                                    sampler=RandomSampler(val_dataset, replacement=True, num_samples=int(1e3)),
                                    shuffle=False, pin_memory=True, batch_size=T.config.batch_size)

            model.eval()
            for batch in tqdm(val_loader):
                batch.to(T.config.device)
                x = batch.squeeze()

                x_hat, z, sample_likelihood, loss = model(x.float())
                losses.append(loss.item())

                x_hat_bin = to_binary(x_hat)
                batch_baccs = [ClassificationMetrics(x[i], x_hat_bin[i]).balanced_accuracy() for i in range(len(x))]
                balanced_accuracies.extend(batch_baccs)

            model.train()

            mean_val_loss = sum(losses) / len(losses)
            mean_balanced_accuracies = sum(balanced_accuracies) / len(balanced_accuracies)

            trainer.history['iter_num'].append(trainer.iter_num)
            trainer.history['train_loss'].append(trainer.loss.item())
            trainer.history['val_loss'].append(mean_val_loss)
            trainer.history['val_ba'].append(mean_balanced_accuracies)

            print(f"Iter: {trainer.iter_num}, train loss: {round(trainer.loss.item(), 4)}, "
                  f"val loss: {round(mean_val_loss, 4)}, balanced accuracy: {round(mean_balanced_accuracies, 4)}")

            history_path = os.path.join(trainer.config.out_path, f"training_history.csv")
            pd.DataFrame(trainer.history).to_csv(history_path)

    T.set_callback('on_batch_end', batch_end_callback)
    T.run()

