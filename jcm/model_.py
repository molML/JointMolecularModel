
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch import Tensor
import torch.distributions
import numpy as np
from tqdm import tqdm
from tqdm.auto import trange
from sklearn.metrics import jaccard_score
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

#
# class Decoder(nn.Module):
#     def __init__(self, latent_dims):
#         super(Decoder, self).__init__()
#         self.linear1 = nn.Linear(latent_dims, 512)
#         self.linear2 = nn.Linear(512, 1024)
#
#     def forward(self, z):
#         z = F.relu(self.linear1(z))
#         z = torch.sigmoid(self.linear2(z))
#         # F.log_softmax(z, 1)
#         return z
#
#
# class VariationalEncoder(nn.Module):
#     def __init__(self, latent_dims):
#         super(VariationalEncoder, self).__init__()
#         self.linear1 = nn.Linear(1024, 512)
#         self.linear2 = nn.Linear(512, latent_dims)
#         self.linear3 = nn.Linear(512, latent_dims)
#
#         self.N = torch.distributions.Normal(0, 1)
#         self.N.loc = self.N.loc
#         self.N.scale = self.N.scale
#         self.kl = 0
#
#     def forward(self, x):
#         x = torch.flatten(x, start_dim=1)
#         x = F.relu(self.linear1(x))
#         mu = self.linear2(x)
#         sigma = torch.exp(self.linear3(x))
#         z = mu + sigma*self.N.sample(mu.shape).to(device)  # reparameterization trick
#         self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()
#
#         return z
#

class PredictionHead(nn.Module):
    def __init__(self, latent_dims):
        super(PredictionHead, self).__init__()
        self.lin1 = nn.Linear(latent_dims, 1024)
        self.lin2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.log_softmax(self.lin2(x), 1)

        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)
        self.pred_head = PredictionHead(latent_dims)
        self.epochs = 10
        self.lr = 3e-4

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        y_hat = self.pred_head(z)

        return x_hat, y_hat


class JointVAE(torch.nn.Module):

    def __init__(self):
        super(JointVAE, self).__init__()
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_type)
        self.beta = 0.001
        self.loss_fn_x = torch.nn.BCELoss()
        self.loss_fn_y = torch.nn.NLLLoss()

        self.model = VariationalAutoencoder(128)

        # Move the whole model to the gpu
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.lr)

        self.train_loss = []
        self.epochs, self.epoch = self.model.epochs, 0

    def train(self, dataloader: DataLoader, epochs: int = None, verbose: bool = True) -> None:

        bar = trange(self.epochs if epochs is None else epochs, disable=not verbose)
        scaler = torch.cuda.amp.GradScaler()

        for _ in bar:
            running_loss = 0
            items = 0

            for idx, batch in enumerate(dataloader):
                self.optimizer.zero_grad()

                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    # break
                    x = batch[0].to(device)  # GPU
                    y = batch[1].to(device)

                    x_hat, y_hat = self.model(x)

                    if len(y_hat) == 0:
                        y_hat = y_hat.unsqueeze(0)

                    reconstruction_loss = self.loss_fn_x(x_hat, x)
                    kl_loss = self.model.encoder.kl / x.shape[0]
                    class_loss = self.loss_fn_y(y_hat, y.squeeze())

                    loss = reconstruction_loss + self.beta * kl_loss + class_loss

                scaler.scale(loss).backward()
                # loss.backward()
                scaler.step(self.optimizer)
                # self.optimizer.step()
                scaler.update()

                running_loss += loss.item()
                items += len(y)

        epoch_loss = running_loss / items
        bar.set_postfix(loss=f'{epoch_loss:.4f}')
        self.train_loss.append(epoch_loss)
        self.epoch += 1

    def predict(self, dataloader: DataLoader, n_sample: int = 10) -> Tensor:
        """ Predict

        :param dataloader: Torch geometric data loader with data
        :return: A 1D-tensors
        """
        y_hats = torch.tensor([]).to(self.device)
        x_hats = torch.tensor([]).to(self.device)
        x_nnls = torch.tensor([]).to(self.device)
        with torch.no_grad():
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                for batch in dataloader:
                    y_hat_batch, x_hat_batch, x_nnl_batch = [], [], []
                    x = batch[0].to(self.device)
                    for i in range(n_sample):
                        x_hat, y_hat = self.model(x)
                        x_hat_nnl = nnl_ecfps(x_hat, x)
                        y_hat_batch.append(y_hat)
                        x_hat_batch.append(x_hat)
                        x_nnl_batch.append(x_hat_nnl)

                    y_hats = torch.cat((y_hats, torch.stack(y_hat_batch, 1)), 0)
                    x_hats = torch.cat((x_hats, torch.stack(x_hat_batch, 1)), 0)
                    x_nnls = torch.cat((x_nnls, torch.stack(x_nnl_batch, 1)), 0)

        # fix potential -inf values in the NLLs
        x_nnls[x_nnls == float('-inf')] = torch.min(x_nnls[x_nnls != float('-inf')])

        return y_hats, x_hats, x_nnls

def nnl_ecfps(x_hat, x):
    """ take the sample-wise sum of the log of the class-specific probabilities for the true class to get the
    sample-wise negative log likelihood

    :param x_hat: Tensor of ecfp bit probabilities with shape (n, h)
    :param x:  Binary Tensor with shape (n, h)
    :return: Tensor shape (n) with the total NNL for each sample
    """
    x_hat_c = torch.stack((1 - x_hat, x_hat), 2)
    batch_dim, sample_dim = x_hat_c.shape[0], x_hat_c.shape[1]
    class_specific_probabilities = x_hat_c[
        torch.arange(batch_dim).unsqueeze(1), torch.arange(sample_dim).unsqueeze(0), x.long()]
    total_nnl = torch.sum(torch.log(class_specific_probabilities), 1)

    return total_nnl



########


import pandas as pd
df = pd.read_csv('data/moleculeace/CHEMBL234_Ki.csv')

df_train = df[df['split'] == 'train']
df_test = df[df['split'] == 'test']


from rdkit.Chem import Descriptors
from rdkit import Chem

descr = Descriptors.CalcMolDescriptors(Chem.MolFromSmiles('c1ccccc1'))
', '.join(list(descr.keys()))

ds_test = MasterDataset('test', representation='ecfp', dataset='ALDH1')
x_test, y_test, smiles_test = ds_test.all()
# test_loader = to_torch_dataloader(x_test, y_test, batch_size=128)

class_weights = [1 - sum((y_test == 0) * 1) / len(y_test), 1 - sum((y_test == 1) * 1) / len(y_test)]
weights = [class_weights[i] for i in y_test]
sampler = WeightedRandomSampler(weights, num_samples=len(y_test), replacement=True)

test_loader = to_torch_dataloader(x_test, y_test, batch_size=128, sampler=sampler)


model = JointVAE()
model.train(test_loader)
y_hats, x_hats, x_nnls = model.predict(test_loader, n_sample=100)
