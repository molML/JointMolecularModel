
import os
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler
from jcm.utils import to_binary, ClassificationMetrics
from jcm.datasets import MoleculeDataset
from jcm.model import VAE, JVAE
from jcm.trainer import Trainer
from jcm.config import Config

if __name__ == '__main__':

    df = pd.read_csv('data/moleculeace/CHEMBL234_Ki.csv')
    df_train = df[df['split'] == 'train']
    df_test = df[df['split'] == 'test']

    smiles_train = df_train.smiles.tolist()
    y_train = to_binary(torch.tensor(df_train['exp_mean [nM]'].tolist()), threshold=100)
    train_dataset = MoleculeDataset(smiles_train, y_train, descriptor='ecfp')

    smiles_val = df_test.smiles.tolist()
    y_val = to_binary(torch.tensor(df_test['exp_mean [nM]'].tolist()), threshold=100)
    val_dataset = MoleculeDataset(smiles_val, y_val, descriptor='ecfp')

    hyperparameters = {'input_dim': 1024,
                       "latent_dim": 128,
                       "hidden_dim_vae": 1024,
                       'out_dim_vae': 1024,
                       'beta': 0.001,
                       "n_layers_mlp": 2,
                       "hidden_dim_mlp": 1024,
                       "anchored": True,
                       "l2_lambda": 1e-4,
                       "n_ensemble": 10,
                       'class_scaling_factor': 20,
                       'lr': 3e-4}

    config = Config(max_iters=1000, batch_size=128)
    config.set_hyperparameters(**hyperparameters)

    model = JVAE(**config.hyperparameters)
    model.vae.load_state_dict(torch.load('/results/chembl_vae/pretrained_vae.pt'))
    T = Trainer(config, model, train_dataset)

    def batch_end_callback(trainer):
        if trainer.iter_num % 10 == 0 and trainer.iter_num > 0:
            balanced_accuracies = []
            losses = []

            # ckpt_path = os.path.join(trainer.config.out_path, f"pretrained_vae_{trainer.iter_num}.pt")
            # torch.save(model.state_dict(), ckpt_path)

            val_loader = DataLoader(val_dataset,
                                    sampler=RandomSampler(val_dataset, replacement=True, num_samples=int(1e3)),
                                    shuffle=False, pin_memory=True, batch_size=trainer.config.batch_size)

            model.eval()
            for x, y in tqdm(val_loader):

                x.to(trainer.config.device)
                y.to(trainer.config.device)
                x = x.squeeze().float()
                y = y.squeeze()

                y_hat, z, sample_likelihood, loss = model(x, y)
                losses.append(loss.item())

                # x_hat_bin = to_binary(y_hat)
                # batch_baccs = [ClassificationMetrics(x[i], x_hat_bin[i]).balanced_accuracy() for i in range(len(x))]
                # balanced_accuracies.extend(batch_baccs)

            model.train()

            mean_val_loss = sum(losses) / len(losses)
            print(trainer.iter_num, trainer.loss.item(), mean_val_loss)
            # mean_balanced_accuracies = sum(balanced_accuracies) / len(balanced_accuracies)

            # trainer.history['iter_num'].append(trainer.iter_num)
            # trainer.history['train_loss'].append(trainer.loss.item())
            # trainer.history['val_loss'].append(mean_val_loss)
            # trainer.history['val_ba'].append(mean_balanced_accuracies)
            #
            # print(f"Iter: {trainer.iter_num}, train loss: {round(trainer.loss.item(), 4)}, "
            #       f"val loss: {round(mean_val_loss, 4)}, balanced accuracy: {round(mean_balanced_accuracies, 4)}")
            #
            # history_path = os.path.join(trainer.config.out_path, f"training_history2.csv")
            # pd.DataFrame(trainer.history).to_csv(history_path)

    T.set_callback('on_batch_end', batch_end_callback)
    T.run()
