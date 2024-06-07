
import torch
from torch import nn
from torch import Tensor
from torch import functional as F
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader
from dataprep.descriptors import encoding_to_smiles
from jcm.utils import single_batchitem_fix
from jcm.modules.lstm import AutoregressiveLSTM
from jcm.modules.utils import BaseModule
from jcm.modules.cnn import CnnEncoder
from jcm.modules.mlp import Ensemble
from jcm.modules.variational import VariationalEncoder
from constants import VOCAB


class DeNovoLSTM(AutoregressiveLSTM, BaseModule):
    # SMILES -> LSTM -> SMILES

    def __init__(self, config):
        self.config = config
        super(DeNovoLSTM, self).__init__(**self.config.hyperparameters)

    @BaseModule().inference
    def generate(self, n: int = 1000, design_length: int = 102, batch_size: int = 256, temperature: int = 1,
                 sample: bool = True):

        # chunk up n designs into batches (i.e., [400, 400, 200] for n=1000 and batch_size=400)
        chunks = [batch_size] * (n // batch_size) + ([n % batch_size] if n % batch_size else [])
        all_designs = []

        for chunk in chunks:
            # init start tokens and add them to the list of generated tokens
            current_token = self.init_start_tokens(batch_size=chunk)
            tokens = [current_token.squeeze()]

            # init an empty hidden and cell state for the first token
            hidden_state, cell_state = self.init_hidden(batch_size=chunk)

            # For every 'current token', generate the next one
            for t_i in range(design_length - 1):  # loop over all tokens in the sequence

                # Get the SMILES embeddings
                x_i = self.embedding_layer(current_token)

                # next token prediction
                x_hat, (hidden_state, cell_state) = self.lstm(x_i, (hidden_state, cell_state))
                logits = F.relu(self.fc(x_hat))

                # perform temperature scaling
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)

                # Get the next token
                if sample:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    _, next_token = torch.topk(probs, k=1, dim=-1)

                # update the 'current token' and the list of generated tokens
                tokens.append(next_token.squeeze())
                current_token = next_token

            tokens = torch.stack(tokens, 1) if n > 1 else torch.stack(tokens).unsqueeze(0)
            smiles = encoding_to_smiles(tokens)
            all_designs.extend(smiles)

        return all_designs

    @BaseModule().inference
    def predict(self, dataset, batch_size: int = 256, sample: bool = False):
        """ Get predictions from a dataset

           :param dataset: dataset of the data to predict; jcm.datasets.MoleculeDataset
           :param batch_size: prediction batch size (default=256)
           :param sample: toggles sampling from the dataset (e.g. for callbacks where you don't full dataset inference)

           :return: token probabilities (n x sequence length x vocab size),
                    embeddings (n x sequence length x embedding size),
                    sample losses (n)
        """

        val_loader = get_val_loader(self.config, dataset, batch_size, sample)

        all_probs = []
        all_embeddings = []
        all_sample_losses = []

        for x in val_loader:

            # predict
            probs, embeddings, sample_losses, loss = self(x.long().to(self.device))

            all_probs.append(probs)
            all_embeddings.append(embeddings)
            all_sample_losses.append(sample_losses)

        all_probs = torch.cat(all_probs, 0)
        all_embeddings = torch.cat(all_embeddings)
        all_sample_losses = torch.cat(all_sample_losses, 0)

        return all_probs, all_embeddings, all_sample_losses

    def init_start_tokens(self, batch_size: int):
        x = torch.zeros((batch_size, 1), device=self.device).long()
        x[:, 0] = VOCAB['start_idx']

        return x


class EcfpMLP(nn.Module):
    # ECFP -> MLP -> property
    def __init__(self):
        super(EcfpMLP, self).__init__()

    @torch.no_grad()
    def predict(self):
        pass

    @staticmethod
    def callback():
        pass


class SmilesMLP(nn.Module):
    # smiles -> CNN -> variational -> MLP -> property
    def __init__(self):
        super(SmilesMLP, self).__init__()

    @torch.no_grad()
    def predict(self):
        pass

    @staticmethod
    def callback():
        pass


class VAE(nn.Module):
    # SMILES -> CNN -> variational -> LSTM -> SMILES
    @torch.no_grad()
    def generate(self):
        pass

    @torch.no_grad()
    def predict(self):
        pass

    @staticmethod
    def callback():
        pass


class JointChemicalModel(nn.Module):
    # SMILES -> CNN -> variational -> LSTM -> SMILES
    #                            |
    #                           MLP -> property
    @torch.no_grad()
    def generate(self):
        pass

    @torch.no_grad()
    def predict(self):
        pass

    @staticmethod
    def callback():
        pass

def get_val_loader(config, dataset, batch_size, sample):
    if sample:
        num_samples = config.val_molecules_to_sample
        val_loader = DataLoader(dataset,
                                sampler=RandomSampler(dataset, replacement=True, num_samples=num_samples),
                                shuffle=False, pin_memory=True, batch_size=batch_size,
                                collate_fn=single_batchitem_fix)
    else:
        val_loader = DataLoader(dataset, sampler=None, shuffle=False, pin_memory=True, batch_size=batch_size,
                                collate_fn=single_batchitem_fix)

    return val_loader
