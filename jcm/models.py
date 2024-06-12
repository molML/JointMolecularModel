
import torch
from torch import nn
from torch import Tensor
from torch import functional as F

from cheminformatics.encoding import encoding_to_smiles
from jcm.utils import get_val_loader, batch_management
from jcm.modules.lstm import AutoregressiveLSTM, init_start_tokens, DecoderLSTM
from jcm.modules.base import BaseModule
from jcm.modules.cnn import CnnEncoder
from jcm.modules.mlp import Ensemble
from jcm.modules.variational import VariationalEncoder
from jcm.datasets import MoleculeDataset
from constants import VOCAB
from cheminformatics.encoding import encoding_to_smiles, strip_smiles, probs_to_smiles


class DeNovoLSTM(AutoregressiveLSTM, BaseModule):
    # SMILES -> LSTM -> SMILES

    def __init__(self, config, **kwargs):
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
            current_token = init_start_tokens(batch_size=chunk, device=self.device)
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
    def predict(self, dataset: MoleculeDataset, batch_size: int = 256, sample: bool = False) -> (Tensor, Tensor, list):
        """ Get predictions from a dataset

           :param dataset: dataset of the data to predict; jcm.datasets.MoleculeDataset
           :param batch_size: prediction batch size (default=256)
           :param sample: toggles sampling from the dataset (e.g. for callbacks where you don't full dataset inference)

           :return: token probabilities (n x sequence length x vocab size),
                    sample losses (n),
                    list of true SMILES strings
        """

        val_loader = get_val_loader(self.config, dataset, batch_size, sample)

        all_probs = []
        all_sample_losses = []
        all_smiles = []

        for x in val_loader:

            x, y = batch_management(x, self.device)

            # reconvert the encoding to smiles and save them. This is inefficient, but due to on the go smiles
            # augmentation it is impossible to get this info from the dataloader directly
            all_smiles.append(encoding_to_smiles(x, strip=True))

            # predict
            probs, sample_losses, loss = self(x)

            all_probs.append(probs)
            all_sample_losses.append(sample_losses)

        all_probs = torch.cat(all_probs, 0)
        all_sample_losses = torch.cat(all_sample_losses, 0)

        return all_probs, all_sample_losses, all_smiles


class VAE(BaseModule):
    # SMILES -> CNN -> variational -> LSTM -> SMILES
    def __init__(self, config, **kwargs):
        super(VAE, self).__init__()

        self.config = config
        self.device = config.hyperparameters['device']
        self.register_buffer('beta', torch.tensor(config.hyperparameters['beta']))

        self.cnn = CnnEncoder(**config.hyperparameters)
        self.variational_layer = VariationalEncoder(var_input_dim=self.cnn.out_dim, **config.hyperparameters)
        self.lstm = DecoderLSTM(**self.config.hyperparameters)

    def forward(self, x: Tensor, y: Tensor = None) -> (Tensor, Tensor, Tensor, Tensor):
        """ Reconstruct a batch of molecule

        :param x: :math:`(N, C)`, batch of integer encoded molecules
        :param y: does nothing, here for compatibility sake
        :return: sequence_probs, z, molecule_loss, loss
        """

        # Embed the integer encoded molecules with the same embedding layer that is used later in the LSTM
        # We transpose it from (batch size x sequence length x embedding) to (batch size x embedding x sequence length)
        # so the embedding is the channel instead of the sequence length
        embedding = self.lstm.embedding_layer(x).transpose(1, 2)

        # Encode the molecule into a latent vector z
        z = self.variational_layer(self.cnn(embedding))

        # Decode z back into a molecule
        sequence_probs, molecule_loss, loss = self.lstm(z, x)

        # Add the KL-divergence loss from the variational layer
        loss_kl = self.variational_layer.kl / x.shape[0]
        loss = loss + self.beta * loss_kl

        return sequence_probs, z, molecule_loss, loss

    @BaseModule().inference
    def generate(self):
        raise NotImplementedError('.generate() function has not been implemented yet')

    @BaseModule().inference
    def predict(self, dataset: MoleculeDataset, batch_size: int = 256, sample: bool = False) -> (Tensor, Tensor, list):
        """ Do inference over molecules in a dataset

        :param dataset: MoleculeDataset that returns a batch of integer encoded molecules :math:`(N, C)`
        :param batch_size: number of samples in a batch
        :param sample: toggles sampling from the dataset, e.g. when doing inference over part of the data for validation
        :return: token_probabilities :math:`(N, S, C)`, where S is sequence length, molecule losses :math:`(N)`, and a
        list of true SMILES strings. Token probabilities do not include the probability for the start token, hence the
        sequence length is reduced by one
        """

        val_loader = get_val_loader(self.config, dataset, batch_size, sample)

        all_probs = []
        all_molecule_losses = []
        all_smiles = []

        for x in val_loader:
            x, y = batch_management(x, self.device)

            # reconvert the encoding to smiles and save them. This is inefficient, but due to on the go smiles
            # augmentation it is impossible to get this info from the dataloader directly
            all_smiles.extend(encoding_to_smiles(x, strip=True))

            # predict
            sequence_probs, z, molecule_loss, loss = self(x)

            all_probs.append(sequence_probs)
            all_molecule_losses.append(molecule_loss)

        all_probs = torch.cat(all_probs, 0)
        all_molecule_losses = torch.cat(all_molecule_losses, 0)

        return all_probs, all_molecule_losses, all_smiles

    @BaseModule().inference
    def get_z(self, dataset: MoleculeDataset, batch_size: int = 256) -> (Tensor, list):
        """ Get the latent representation :math:`z` of molecules

        :param dataset: MoleculeDataset that returns a batch of integer encoded molecules :math:`(N, C)`
        :param batch_size: number of samples in a batch
        :return: latent vectors :math:`(N, H)`, where hidden is the VAE compression dimension
        """

        val_loader = get_val_loader(self.config, dataset, batch_size)

        all_z = []
        all_smiles = []
        for x in val_loader:
            x, y = batch_management(x, self.device)
            all_smiles.extend(encoding_to_smiles(x, strip=True))
            sequence_probs, z, molecule_loss, loss = self(x)
            all_z.append(z)

        return torch.cat(all_z), all_smiles


class SmilesMLP(BaseModule):
    # SMILES -> CNN -> variational -> MLP -> y
    def __init__(self, config, **kwargs):
        super(SmilesMLP, self).__init__()

        self.config = config
        self.device = config.hyperparameters['device']
        self.register_buffer('beta', torch.tensor(config.hyperparameters['beta']))

        self.embedding_layer = nn.Embedding(num_embeddings=config.hyperparameters['vocabulary_size'],
                                            embedding_dim=config.hyperparameters['token_embedding_dim'])
        self.cnn = CnnEncoder(**config.hyperparameters)
        self.variational_layer = VariationalEncoder(var_input_dim=self.cnn.out_dim, **config.hyperparameters)
        self.mlp = Ensemble(**config.hyperparameters)

    def forward(self, x: Tensor, y: Tensor = None) -> (Tensor, Tensor, Tensor, Tensor):
        """ Reconstruct a batch of molecule

        :param x: :math:`(N, C)`, batch of integer encoded molecules
        :param y: :math:`(N)`, labels, optional. When None, no loss is computed (default=None)
        :return: sequence_probs, z, loss
        """

        # Embed the integer encoded molecules with the same embedding layer that is used later in the LSTM
        # We transpose it from (batch size x sequence length x embedding) to (batch size x embedding x sequence length)
        # so the embedding is the channel instead of the sequence length
        embedding = self.embedding_layer(x).transpose(1, 2)

        # Encode the molecule into a latent vector z
        z = self.variational_layer(self.cnn(embedding))

        # Predict a property from this embedding
        y_logits_N_K_C, molecule_loss, loss = self.mlp(z, y)

        # Add the KL-divergence loss from the variational layer
        if loss is not None:
            loss_kl = self.variational_layer.kl / x.shape[0]
            loss = loss + self.beta * loss_kl

        return y_logits_N_K_C, z, loss

    @BaseModule().inference
    def generate(self):
        raise NotImplementedError('.generate() function does not apply to this predictive model yet')

    @BaseModule().inference
    def predict(self, dataset: MoleculeDataset, batch_size: int = 256, sample: bool = False) -> \
            (Tensor, Tensor, Tensor):
        """ Do inference over molecules in a dataset

        :param dataset: MoleculeDataset that returns a batch of integer encoded molecules :math:`(N, C)`
        :param batch_size: number of samples in a batch
        :param sample: toggles sampling from the dataset, e.g. when doing inference over part of the data for validation
        :return: token_probabilities :math:`(N, K, C)`, where K is ensemble size, loss, and target labels :math:`(N)`.
        """

        val_loader = get_val_loader(self.config, dataset, batch_size, sample)

        all_logits = []
        all_ys = []
        all_losses = []

        for x in val_loader:
            x, y = batch_management(x, self.device)

            # predict
            y_logits_N_K_C, z, loss = self(x, y)

            all_logits.append(y_logits_N_K_C)
            if y is not None:
                all_losses.append(loss)
                all_ys.append(y)

        all_logits = torch.cat(all_logits, 0)
        all_ys = torch.cat(all_ys) if len(all_ys) > 0 else None
        all_losses = torch.mean(torch.cat(all_losses)) if len(all_losses) > 0 else None

        return all_logits, all_losses, all_ys

    @BaseModule().inference
    def get_z(self, dataset: MoleculeDataset, batch_size: int = 256) -> (Tensor, list):
        """ Get the latent representation :math:`z` of molecules

        :param dataset: MoleculeDataset that returns a batch of integer encoded molecules :math:`(N, C)`
        :param batch_size: number of samples in a batch
        :return: latent vectors :math:`(N, H)`, where hidden is the VAE compression dimension
        """

        val_loader = get_val_loader(self.config, dataset, batch_size)

        all_z = []
        all_smiles = []
        for x in val_loader:
            x, y = batch_management(x, self.device)
            all_smiles.extend(encoding_to_smiles(x, strip=True))
            y_logits_N_K_C, z, loss = self(x)
            all_z.append(z)

        return torch.cat(all_z), all_smiles


class ECFPMLP(Ensemble, BaseModule):
    # ECFP -> MLP -> yhat

    def __init__(self, config, **kwargs):
        self.config = config
        super(ECFPMLP, self).__init__(**self.config.hyperparameters)

    @BaseModule().inference
    def predict(self, dataset: MoleculeDataset, batch_size: int = 256, sample: bool = False) -> \
            (Tensor, Tensor, Tensor):
        """ Do inference over molecules in a dataset

        :param dataset: MoleculeDataset that returns a batch of integer encoded molecules :math:`(N, C)`
        :param batch_size: number of samples in a batch
        :param sample: toggles sampling from the dataset, e.g. when doing inference over part of the data for validation
        :return: token_probabilities :math:`(N, K, C)`, where K is ensemble size, loss, and target labels :math:`(N)`.
        """

        val_loader = get_val_loader(self.config, dataset, batch_size, sample)

        all_logits = []
        all_ys = []
        all_losses = []

        for x in val_loader:
            x, y = batch_management(x, self.device)

            # predict
            y_logits_N_K_C, _, loss = self(x, y)

            all_logits.append(y_logits_N_K_C)
            if y is not None:
                all_losses.append(loss)
                all_ys.append(y)

        all_logits = torch.cat(all_logits, 0)
        all_ys = torch.cat(all_ys) if len(all_ys) > 0 else None
        all_losses = torch.mean(torch.cat(all_losses)) if len(all_losses) > 0 else None

        return all_logits, all_losses, all_ys



# class JointChemicalModel(nn.Module):
#     # SMILES -> CNN -> variational -> LSTM -> SMILES
#     #                            |
#     #                           MLP -> property
#     @torch.no_grad()
#     def generate(self):
#         pass
#
#     @torch.no_grad()
#     def predict(self):
#         pass
#
#     @staticmethod
#     def callback():
#         pass

