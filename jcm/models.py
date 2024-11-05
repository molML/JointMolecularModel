import warnings
import copy
from collections import OrderedDict
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from jcm.utils import get_val_loader, batch_management, filter_params
from jcm.modules.rnn import RNN, init_start_tokens, ConditionedRNN
from jcm.modules.base import BaseModule
from jcm.modules.cnn import CnnEncoder
from jcm.modules.mlp import Ensemble
from jcm.modules.variational import VariationalEncoder
from jcm.datasets import MoleculeDataset
from jcm.modules.rnn import init_rnn_hidden
from cheminformatics.encoding import encoding_to_smiles, probs_to_smiles


class DeNovoRNN(RNN, BaseModule):
    # SMILES -> RNN -> SMILES

    def __init__(self, config, **kwargs):
        self.config = config
        super(DeNovoRNN, self).__init__(**self.config.hyperparameters)

    def generate(self, n: int = 1000, design_length: int = 102, batch_size: int = 256, temperature: int = 1,
                 sample: bool = True):

        # chunk up n designs into batches (i.e., [400, 400, 200] for n=1000 and batch_size=400)
        chunks = [batch_size] * (n // batch_size) + ([n % batch_size] if n % batch_size else [])
        all_designs = []

        self.eval()
        with torch.no_grad():
            for chunk in chunks:
                # init start tokens and add them to the list of generated tokens
                current_token = init_start_tokens(batch_size=chunk, device=self.device)
                tokens = [current_token.squeeze()]

                # init an empty hidden and cell state for the first token
                hidden_state = init_rnn_hidden(num_layers=self.num_layers, batch_size=chunk, hidden_size=self.hidden_size,
                                               device=self.device, rnn_type=self.rnn_type)

                # For every 'current token', generate the next one
                for t_i in range(design_length - 1):  # loop over all tokens in the sequence

                    # Get the SMILES embeddings
                    x_i = self.embedding_layer(current_token)

                    # next token prediction
                    x_hat, hidden_state = self.rnn(x_i, hidden_state)
                    logits = self.fc(x_hat)

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

    def predict(self, dataset: MoleculeDataset, batch_size: int = 256, sample: bool = False,
                convert_probs_to_smiles: bool = False) -> (Tensor, Tensor, list):
        """ Get predictions from a dataset

           :param dataset: dataset of the data to predict; jcm.datasets.MoleculeDataset
           :param batch_size: prediction batch size (default=256)
           :param convert_probs_to_smiles: toggles if probabilities are converted to SMILES strings right away
           :param sample: toggles sampling from the dataset (e.g. for callbacks where you don't full dataset inference)

           :return: token probabilities (n x sequence length x vocab size),
                    sample losses (n),
                    list of true SMILES strings
        """

        val_loader = get_val_loader(self.config, dataset, batch_size, sample)

        all_probs = []
        all_reconstruction_losses = []
        all_smiles = []

        self.eval()
        with torch.no_grad():
            for x in val_loader:

                x, y = batch_management(x, self.device)

                # reconvert the encoding to smiles and save them. This is inefficient, but due to on the go smiles
                # augmentation it is impossible to get this info from the dataloader directly
                all_smiles.extend(encoding_to_smiles(x, strip=True))

                # predict
                probs, loss = self(x)

                if convert_probs_to_smiles:
                    smiles = probs_to_smiles(probs)
                    all_probs.extend(smiles)
                else:
                    all_probs.append(probs)
                all_reconstruction_losses.append(self.reconstruction_loss)

            if not convert_probs_to_smiles:
                all_probs = torch.cat(all_probs, 0)
            reconstruction_loss = total_loss = torch.cat(all_reconstruction_losses, 0)

            output = {"token_probs_N_S_C": all_probs, "reconstruction_loss": reconstruction_loss,
                      "total_loss": total_loss, "smiles": all_smiles}

        return output


class AE(BaseModule):
    """ SMILES -> CNN -> z -> RNN -> SMILES """
    def __init__(self, config, **kwargs):
        super(AE, self).__init__()

        self.config = config
        self.device = config.device

        self.cnn = CnnEncoder(**config.hyperparameters)
        self.z_layer = nn.Linear(self.cnn.out_dim, self.config.z_size)
        self.rnn = ConditionedRNN(**self.config.hyperparameters)

        self.reconstruction_loss = None
        self.total_loss = None
        self.loss = None

    def forward(self, x: Tensor, y: Tensor = None) -> (Tensor, Tensor, Tensor, Tensor):
        """ Reconstruct a batch of molecule

        :param x: :math:`(N, C)`, batch of integer encoded molecules
        :param y: does nothing, here for compatibility’s sake
        :return: sequence_probs, z, molecule_loss, loss
        """

        # Embed the integer encoded molecules with the same embedding layer that is used later in the rnn
        # We transpose it from (batch size x sequence length x embedding) to (batch size x embedding x sequence length)
        # so the embedding is the channel instead of the sequence length
        embedding = self.rnn.embedding_layer(x).transpose(1, 2)

        # Encode the molecule into a latent vector z
        z = self.z_layer(self.cnn(embedding))

        # Decode z back into a molecule
        sequence_probs, loss = self.rnn(z, x)
        self.loss = loss
        self.reconstruction_loss = self.total_loss = self.rnn.reconstruction_loss

        return sequence_probs, z, self.loss

    def predict(self, dataset: MoleculeDataset, batch_size: int = 256, sample: bool = False,
                convert_probs_to_smiles: bool = False) -> (Tensor, Tensor, list):
        """ Do inference over molecules in a dataset

        :param dataset: MoleculeDataset that returns a batch of integer encoded molecules :math:`(N, C)`
        :param batch_size: number of samples in a batch
        :param convert_probs_to_smiles: toggles if probabilities are converted to SMILES strings right away
        :param sample: toggles sampling from the dataset, e.g. when doing inference over part of the data for validation
        :return: token_probabilities :math:`(N, S, C)`, where S is sequence length, molecule losses :math:`(N)`, and a
        list of true SMILES strings. Token probabilities do not include the probability for the start token, hence the
        sequence length is reduced by one
        """

        val_loader = get_val_loader(self.config, dataset, batch_size, sample)

        all_probs = []
        all_reconstruction_losses = []
        all_total_losses = []
        all_smiles = []

        self.eval()
        with torch.no_grad():
            for x in val_loader:
                x, y = batch_management(x, self.device)

                # reconvert the encoding to smiles and save them. This is inefficient, but due to on the go smiles
                # augmentation it is impossible to get this info from the dataloader directly
                all_smiles.extend(encoding_to_smiles(x, strip=True))

                # predict
                sequence_probs, z, loss = self(x)

                if convert_probs_to_smiles:
                    smiles = probs_to_smiles(sequence_probs)
                    all_probs.extend(smiles)
                else:
                    all_probs.append(sequence_probs)

                all_reconstruction_losses.append(self.reconstruction_loss)
                all_total_losses.append(self.total_loss)

            if not convert_probs_to_smiles:
                all_probs = torch.cat(all_probs, 0)
            reconstruction_loss = torch.cat(all_reconstruction_losses, 0)
            total_loss = torch.cat(all_total_losses, 0)

            output = {"token_probs_N_S_C": all_probs, "reconstruction_loss": reconstruction_loss,
                      "total_loss": total_loss, "smiles": all_smiles}

        return output

    def get_z(self, dataset: MoleculeDataset, batch_size: int = 256) -> (Tensor, list):
        """ Get the latent representation :math:`z` of molecules

        :param dataset: MoleculeDataset that returns a batch of integer encoded molecules :math:`(N, C)`
        :param batch_size: number of samples in a batch
        :return: latent vectors :math:`(N, H)`, where hidden is the VAE compression dimension
        """

        val_loader = get_val_loader(self.config, dataset, batch_size)

        all_z = []
        all_smiles = []

        self.eval()
        with torch.no_grad():
            for x in val_loader:
                x, y = batch_management(x, self.device)
                all_smiles.extend(encoding_to_smiles(x, strip=True))

                embedding = self.rnn.embedding_layer(x).transpose(1, 2)
                # Encode the molecule into a latent vector z
                z = self.z_layer(self.cnn(embedding))
                all_z.append(z)

        return torch.cat(all_z), all_smiles


class VAE(BaseModule):
    # SMILES -> CNN -> variational -> RNN -> SMILES
    def __init__(self, config, **kwargs):
        super(VAE, self).__init__()

        self.config = config
        self.device = config.device
        self.register_buffer('beta', torch.tensor(config.beta))

        self.cnn = CnnEncoder(**config.hyperparameters)
        self.variational_layer = VariationalEncoder(var_input_dim=self.cnn.out_dim, **config.hyperparameters)
        self.rnn = ConditionedRNN(**self.config.hyperparameters)

        self.reconstruction_loss = None
        self.kl_loss = None
        self.total_loss = None
        self.loss = None

    def forward(self, x: Tensor, y: Tensor = None) -> (Tensor, Tensor, Tensor, Tensor):
        """ Reconstruct a batch of molecule

        :param x: :math:`(N, C)`, batch of integer encoded molecules
        :param y: does nothing, here for compatibility’s sake
        :return: sequence_probs, z, molecule_loss, loss
        """

        # Embed the integer encoded molecules with the same embedding layer that is used later in the rnn
        # We transpose it from (batch size x sequence length x embedding) to (batch size x embedding x sequence length)
        # so the embedding is the channel instead of the sequence length
        embedding = self.rnn.embedding_layer(x).transpose(1, 2)

        # Encode the molecule into a latent vector z
        z = self.variational_layer(self.cnn(embedding))

        # Decode z back into a molecule
        sequence_probs, loss = self.rnn(z, x)
        self.reconstruction_loss = self.rnn.reconstruction_loss

        # Add the KL-divergence loss from the variational layer
        self.kl_loss = self.beta * self.variational_layer.kl

        self.total_loss = self.reconstruction_loss + self.kl_loss
        self.loss = self.total_loss.sum() / x.shape[0]

        return sequence_probs, z, self.loss

    def generate(self, z: Tensor = None, seq_length: int = 101, n: int = 1, batch_size: int = 256) -> Tensor:
        """ Generate molecules from either a tensor of latent representations or random tensors

        :param z: Tensor (N, Z)
        :param seq_length: number of tokens to generate
        :param n: number of molecules to generate. Only applies when z = None, else takes the first dim of z as n
        :param batch_size: size of the batches
        :return: Tensor (N, S, C)
        """

        self.eval()
        with torch.no_grad():
            if z is None:
                chunks = [batch_size] * (n // batch_size) + ([n % batch_size] if n % batch_size else [])
                all_probs = []
                for chunk in chunks:
                    # create a random z vector and scale them to the scale used to train the model
                    z_ = torch.rand(chunk, self.rnn.z_size) * self.config.variational_scale
                    all_probs.append(self.rnn.generate_from_z(z_, seq_len=seq_length+1))
            else:
                n = z.size(0)
                chunks = [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]
                all_probs = []
                for chunk in chunks:
                    z_ = z[chunk]
                    all_probs.append(self.rnn.generate_from_z(z_, seq_len=seq_length+1))

        return torch.cat(all_probs)

    def predict(self, dataset: MoleculeDataset, batch_size: int = 256, sample: bool = False,
                convert_probs_to_smiles: bool = False) -> (Tensor, Tensor, list):
        """ Do inference over molecules in a dataset

        :param dataset: MoleculeDataset that returns a batch of integer encoded molecules :math:`(N, C)`
        :param batch_size: number of samples in a batch
        :param convert_probs_to_smiles: toggles if probabilities are converted to SMILES strings right away
        :param sample: toggles sampling from the dataset, e.g. when doing inference over part of the data for validation
        :return: token_probabilities :math:`(N, S, C)`, where S is sequence length, molecule losses :math:`(N)`, and a
        list of true SMILES strings. Token probabilities do not include the probability for the start token, hence the
        sequence length is reduced by one
        """

        val_loader = get_val_loader(self.config, dataset, batch_size, sample)

        all_probs = []
        all_reconstruction_losses = []
        all_kl_losses = []
        all_total_losses = []
        all_smiles = []
        all_losses = []

        self.eval()
        with torch.no_grad():
            for x in val_loader:
                x, y = batch_management(x, self.device)

                # reconvert the encoding to smiles and save them. This is inefficient, but due to on the go smiles
                # augmentation it is impossible to get this info from the dataloader directly
                all_smiles.extend(encoding_to_smiles(x, strip=True))

                # predict
                sequence_probs, z, loss = self(x)

                if convert_probs_to_smiles:
                    smiles = probs_to_smiles(sequence_probs)
                    all_probs.extend(smiles)
                else:
                    all_probs.append(sequence_probs)
                all_reconstruction_losses.append(self.reconstruction_loss)
                all_kl_losses.append(self.kl_loss)
                all_total_losses.append(self.total_loss)
                all_losses.append(self.loss)

            if not convert_probs_to_smiles:
                all_probs = torch.cat(all_probs, 0)

            reconstruction_loss = torch.cat(all_reconstruction_losses, 0)
            kl_loss = torch.cat(all_kl_losses, 0)
            total_loss = torch.cat(all_total_losses, 0)

            output = {"token_probs_N_S_C": all_probs, "reconstruction_loss": reconstruction_loss,
                      "kl_loss": kl_loss, "total_loss": total_loss, "smiles": all_smiles}

        return output

    def get_z(self, dataset: MoleculeDataset, batch_size: int = 256) -> (Tensor, list):
        """ Get the latent representation :math:`z` of molecules

        :param dataset: MoleculeDataset that returns a batch of integer encoded molecules :math:`(N, C)`
        :param batch_size: number of samples in a batch
        :return: latent vectors :math:`(N, H)`, where hidden is the VAE compression dimension
        """

        val_loader = get_val_loader(self.config, dataset, batch_size)

        all_z = []
        all_smiles = []

        self.eval()
        with torch.no_grad():
            for x in val_loader:
                x, y = batch_management(x, self.device)
                all_smiles.extend(encoding_to_smiles(x, strip=True))

                embedding = self.rnn.embedding_layer(x).transpose(1, 2)
                # Encode the molecule into a latent vector z
                z = self.variational_layer(self.cnn(embedding))
                all_z.append(z)

        return torch.cat(all_z), all_smiles


class SmilesMLP(BaseModule):
    """ SMILES -> CNN -> z -> MLP -> y """
    def __init__(self, config, **kwargs):
        super(SmilesMLP, self).__init__()

        self.config = config
        self.device = config.hyperparameters['device']

        self.embedding_layer = nn.Embedding(num_embeddings=config.hyperparameters['vocabulary_size'],
                                            embedding_dim=config.hyperparameters['token_embedding_dim'])
        self.cnn = CnnEncoder(**config.hyperparameters)
        self.z_layer = nn.Linear(self.cnn.out_dim, self.config.z_size)
        self.mlp = Ensemble(**config.hyperparameters)

        self.loss = None
        self.prediction_loss = None
        self.total_loss = None

    def forward(self, x: Tensor, y: Tensor = None) -> (Tensor, Tensor, Tensor, Tensor):
        """ Reconstruct a batch of molecule

        :param x: :math:`(N, C)`, batch of integer encoded molecules
        :param y: :math:`(N)`, labels, optional. When None, no loss is computed (default=None)
        :return: sequence_probs, z, loss
        """

        # Embed the integer encoded molecules with the same embedding layer that is used later in the rnn
        # We transpose it from (batch size x sequence length x embedding) to (batch size x embedding x sequence length)
        # so the embedding is the channel instead of the sequence length
        embedding = self.embedding_layer(x).transpose(1, 2)

        # Encode the molecule into a latent vector z
        z = self.z_layer(self.cnn(embedding))

        # Predict a property from this embedding
        y_logprobs_N_K_C, self.loss = self.mlp(z, y)
        self.prediction_loss = self.mlp.prediction_loss

        return y_logprobs_N_K_C, z, self.loss

    def generate(self):
        raise NotImplementedError('.generate() function does not apply to this predictive model yet')

    def predict(self, dataset: MoleculeDataset, batch_size: int = 256, sample: bool = False) -> \
            (Tensor, Tensor, Tensor):
        """ Do inference over molecules in a dataset

        :param dataset: MoleculeDataset that returns a batch of integer encoded molecules :math:`(N, C)`
        :param batch_size: number of samples in a batch
        :param sample: toggles sampling from the dataset, e.g. when doing inference over part of the data for validation
        :return: class log probs :math:`(N, K, C)`, where K is ensemble size, loss, and target labels :math:`(N)`
        """

        val_loader = get_val_loader(self.config, dataset, batch_size, sample)

        all_y_logprobs_N_K_C = []
        all_ys = []
        all_prediction_losses = []

        self.eval()
        with torch.no_grad():
            for x in val_loader:
                x, y = batch_management(x, self.device)

                # predict
                y_logprobs_N_K_C, z, loss = self(x, y)

                all_y_logprobs_N_K_C.append(y_logprobs_N_K_C)
                if y is not None:
                    all_prediction_losses.append(self.prediction_loss)
                    all_ys.append(y)

            all_y_logprobs_N_K_C = torch.cat(all_y_logprobs_N_K_C, 0)
            all_ys = torch.cat(all_ys) if len(all_ys) > 0 else None
            prediction_loss = torch.cat(all_prediction_losses) if len(all_prediction_losses) > 0 else None
            total_loss = prediction_loss

            output = {"y_logprobs_N_K_C": all_y_logprobs_N_K_C, "total_loss": total_loss,
                      "prediction_loss": prediction_loss, "y": all_ys}

        return output

    def get_z(self, dataset: MoleculeDataset, batch_size: int = 256) -> (Tensor, list):
        """ Get the latent representation :math:`z` of molecules

        :param dataset: MoleculeDataset that returns a batch of integer encoded molecules :math:`(N, C)`
        :param batch_size: number of samples in a batch
        :return: latent vectors :math:`(N, H)`, where hidden is the VAE compression dimension
        """

        val_loader = get_val_loader(self.config, dataset, batch_size)

        all_z = []
        all_smiles = []

        self.eval()
        with torch.no_grad():
            for x in val_loader:
                x, y = batch_management(x, self.device)
                all_smiles.extend(encoding_to_smiles(x, strip=True))
                y_logprobs_N_K_C, z, loss = self(x)
                all_z.append(z)

        return torch.cat(all_z), all_smiles


class SmilesVarMLP(BaseModule):
    """ SMILES -> CNN -> variational -> MLP -> y """
    def __init__(self, config, **kwargs):
        super(SmilesVarMLP, self).__init__()

        self.config = config
        self.device = config.device
        self.register_buffer('beta', torch.tensor(config.beta))

        self.embedding_layer = nn.Embedding(num_embeddings=config.vocabulary_size,
                                            embedding_dim=config.token_embedding_dim)
        self.cnn = CnnEncoder(**config.hyperparameters)
        self.variational_layer = VariationalEncoder(var_input_dim=self.cnn.out_dim, **config.hyperparameters)
        self.mlp = Ensemble(**config.hyperparameters)

        self.prediction_loss = None
        self.total_loss = None
        self.kl_loss = None
        self.loss = None

    def forward(self, x: Tensor, y: Tensor = None) -> (Tensor, Tensor, Tensor, Tensor):
        """ Reconstruct a batch of molecule

        :param x: :math:`(N, C)`, batch of integer encoded molecules
        :param y: :math:`(N)`, labels, optional. When None, no loss is computed (default=None)
        :return: sequence_probs, z, loss
        """

        # Embed the integer encoded molecules with the same embedding layer that is used later in the rnn
        # We transpose it from (batch size x sequence length x embedding) to (batch size x embedding x sequence length)
        # so the embedding is the channel instead of the sequence length
        embedding = self.embedding_layer(x).transpose(1, 2)

        # Encode the molecule into a latent vector z
        z = self.variational_layer(self.cnn(embedding))

        # Predict a property from this embedding
        y_logprobs_N_K_C, self.loss = self.mlp(z, y)
        self.prediction_loss = self.mlp.prediction_loss

        # Add the KL-divergence loss from the variational layer
        if self.loss is not None:
            self.kl_loss = self.beta * self.variational_layer.kl  # / x.shape[0]

            self.total_loss = self.prediction_loss + self.kl_loss
            self.loss = self.total_loss.sum() / x.shape[0]

        return y_logprobs_N_K_C, z, self.loss

    def generate(self):
        raise NotImplementedError('.generate() function does not apply to this predictive model yet')

    def predict(self, dataset: MoleculeDataset, batch_size: int = 256, sample: bool = False) -> \
            (Tensor, Tensor, Tensor):
        """ Do inference over molecules in a dataset

        :param dataset: MoleculeDataset that returns a batch of integer encoded molecules :math:`(N, C)`
        :param batch_size: number of samples in a batch
        :param sample: toggles sampling from the dataset, e.g. when doing inference over part of the data for validation
        :return: class log probs :math:`(N, K, C)`, where K is ensemble size, loss, and target labels :math:`(N)`
        """

        val_loader = get_val_loader(self.config, dataset, batch_size, sample)

        all_y_logprobs_N_K_C = []
        all_ys = []
        all_prediction_losses = []
        all_kl_losses = []
        all_total_losses = []

        self.eval()
        with torch.no_grad():
            for x in val_loader:
                x, y = batch_management(x, self.device)

                # predict
                y_logprobs_N_K_C, z, loss = self(x, y)

                all_y_logprobs_N_K_C.append(y_logprobs_N_K_C)
                if y is not None:
                    all_prediction_losses.append(self.prediction_loss)
                    all_kl_losses.append(self.kl_loss)
                    all_total_losses.append(self.total_loss)
                    all_ys.append(y)

            all_y_logprobs_N_K_C = torch.cat(all_y_logprobs_N_K_C, 0)
            all_ys = torch.cat(all_ys) if len(all_ys) > 0 else None
            prediction_loss = torch.cat(all_prediction_losses) if len(all_prediction_losses) > 0 else None
            kl_loss = torch.cat(all_kl_losses) if len(all_kl_losses) > 0 else None
            total_loss = torch.cat(all_total_losses) if len(all_total_losses) > 0 else None

            output = {"y_logprobs_N_K_C": all_y_logprobs_N_K_C, "prediction_loss": prediction_loss, "kl_loss": kl_loss,
                      "total_loss": total_loss, "y": all_ys}

        return output

    def get_z(self, dataset: MoleculeDataset, batch_size: int = 256) -> (Tensor, list):
        """ Get the latent representation :math:`z` of molecules

        :param dataset: MoleculeDataset that returns a batch of integer encoded molecules :math:`(N, C)`
        :param batch_size: number of samples in a batch
        :return: latent vectors :math:`(N, H)`, where hidden is the VAE compression dimension
        """

        val_loader = get_val_loader(self.config, dataset, batch_size)

        all_z = []
        all_smiles = []

        self.eval()
        with torch.no_grad():
            for x in val_loader:
                x, y = batch_management(x, self.device)
                all_smiles.extend(encoding_to_smiles(x, strip=True))
                y_logprobs_N_K_C, z, loss = self(x)
                all_z.append(z)

        return torch.cat(all_z), all_smiles


class MLP(Ensemble, BaseModule):
    # ECFP -> MLP -> yhat

    def __init__(self, config, **kwargs):
        self.config = config
        super(MLP, self).__init__(**self.config.hyperparameters)

    def predict(self, dataset: MoleculeDataset, batch_size: int = 256, sample: bool = False) -> \
            (Tensor, Tensor, Tensor):
        """ Do inference over molecules in a dataset

        :param dataset: MoleculeDataset that returns a batch of integer encoded molecules :math:`(N, C)`
        :param batch_size: number of samples in a batch
        :param sample: toggles sampling from the dataset, e.g. when doing inference over part of the data for validation
        :return: y_logprobs_N_K_C :math:`(N, K, C)`, where K is ensemble size, loss, and target labels :math:`(N)`.
        """

        val_loader = get_val_loader(self.config, dataset, batch_size, sample)

        all_y_logprobs_N_K_C = []
        all_ys = []
        all_prediction_losses = []

        self.eval()
        with torch.no_grad():
            for x in val_loader:
                x, y = batch_management(x, self.device)

                # predict
                y_logprobs_N_K_C, loss = self(x, y)

                all_y_logprobs_N_K_C.append(y_logprobs_N_K_C)
                if y is not None:
                    all_prediction_losses.append(self.prediction_loss)
                    all_ys.append(y)

            all_y_logprobs_N_K_C = torch.cat(all_y_logprobs_N_K_C, 0)
            all_ys = torch.cat(all_ys) if len(all_ys) > 0 else None
            prediction_loss = total_loss = torch.cat(all_prediction_losses) if len(all_prediction_losses) > 0 else None

            output = {"y_logprobs_N_K_C": all_y_logprobs_N_K_C, "total_loss": total_loss,
                      "prediction_loss": prediction_loss, "y": all_ys}

        return output


class JMM(BaseModule):
    """ SMILES -> CNN -> (variational) Z -> rnn -> SMILES
                                |
                               MLP -> property
    """
    def __init__(self, config, **kwargs):
        self.config = config
        self.device = self.config.hyperparameters['device']
        super(JMM, self).__init__()
        self.variational = self.config.hyperparameters['variational']
        self.pretrained_ae_path = self.config.hyperparameters['pretrained_ae_path']
        self.pretrained_encoder_mlp_path = self.config.hyperparameters['pretrained_encoder_mlp_path']

        self.ae = VAE(config) if self.variational else AE(config)
        self.mlp = MLP(config)
        self.pretrained_decoder = None

        self.register_buffer('mlp_loss_scalar', torch.tensor(config.hyperparameters['mlp_loss_scalar']))

        self.prediction_loss = None
        self.reconstruction_loss = None
        self.pretrained_decoder_reconstruction_loss = None
        self.kl_loss = None
        self.total_loss = None
        self.loss = None

        self.load_pretrained()

    def load_pretrained(self):
        if self.pretrained_ae_path is not None:
            enc_dec_state_dict = torch.load(self.pretrained_ae_path, map_location=torch.device(self.device))
            self.ae.load_state_dict(enc_dec_state_dict)
            print('Loaded pretrained (V)AE')

        if self.pretrained_encoder_mlp_path is not None:
            enc_mlp = torch.load(self.pretrained_encoder_mlp_path, map_location=torch.device('cpu'))

            self.mlp = enc_mlp.mlp
            self.mlp.device = self.device
            print('Loaded pretrained MLP')

            self.ae.embedding_layer = enc_mlp.embedding_layer
            self.ae.cnn = enc_mlp.cnn

            if self.variational:
                self.ae.variational_layer = enc_mlp.variational_layer
                self.ae.variational_layer.device = self.device
                self.ae.variational_layer.prior_std = self.config.variational_scale
            else:
                self.ae.z_layer = enc_mlp.z_layer
                self.ae.z_layer.device = self.device
            print('Using the encoder from the pretrained SMILES MLP')

            self.pretrained_decoder = copy.deepcopy(self.ae.rnn)
            print('Stored the pretrained decoder to debias OOD scores later')

    # def ood_score(self, dataset: MoleculeDataset, batch_size: int = 256) -> tuple[list[str], Tensor]:
    #     """
    #
    #     :param dataset: MoleculeDataset that returns a batch of integer encoded molecules :math:`(N, C)`
    #     :param batch_size: number of samples in a batch
    #     :return: List of SMILES strings and their (debiased) reconstruction loss
    #     """
    #     if self.pretrained_ae is None:
    #         warnings.warn("No pretrained model is loaded. Load one with the 'load_pretrained' method of this class if "
    #                       "you want to debias your reconstructions")
    #
    #     # reconstruction loss on the finetuned vae
    #     all_reconstruction_losses = []
    #     all_ood_scores = []
    #     all_ood_scores = []
    #     all_smiles = []
    #
    #     val_loader = get_val_loader(self.config, dataset, batch_size, sample=False)
    #
    #     self.eval()
    #     with torch.no_grad():
    #         for x in val_loader:
    #             x, y = batch_management(x, self.device)
    #
    #             # reconvert the encoding to smiles and save them. This is inefficient, but due to on the go smiles
    #             # augmentation it is impossible to get this info from the dataloader directly
    #             all_smiles.extend(encoding_to_smiles(x, strip=True))
    #
    #             # predict
    #             self.forward(x, y)
    #             ood_score = self.reconstruction_loss
    #
    #             # if there's a pretrained model loaded, use it to debias the reconstruction loss
    #             if self.pretrained_ae is not None:
    #                 ood_score_pt = self.pretrained_decoder.reconstruction_loss
    #                 ood_score = ood_score - ood_score_pt
    #
    #             ood_scores.append(ood_score)
    #
    #         output = {"reconstruction_loss": reconstruction_loss,
    #                   "pretrained_reconstruction_loss": pretrained_reconstruction_loss,
    #                   "ood_score": ood_score,
    #                   "smiles": all_smiles}
    #
    #     return output

    def forward(self, x: Tensor, y: Tensor = None) -> (Tensor, Tensor, Tensor, Tensor):
        """ Reconstruct a batch of molecule

        :param x: :math:`(N, C)`, batch of integer encoded molecules
        :param y: :math:`(N)`, target labels
        :return: token_probabilities :math:`N, S, C`, where S is the sequence length. This will be one shorter than the
        input sequence because the start token is not predicted,
                y_logits :math:`(N, K, C)`, where K is ensemble size,
                latent vectors :math:`(N, H)`, where hidden is the VAE compression dimension,
                molecule_loss, loss
        """

        # Reconstruct molecule
        sequence_probs, z, vae_loss = self.ae(x)

        if self.pretrained_decoder is not None:
            self.pretrained_decoder(z, x)

        # predict property from latent representation
        logprobs_N_K_C, mlp_loss = self.mlp(z, y)

        # combine losses, but if y is None, return the loss as None
        if mlp_loss is None:
            self.loss = None
        else:
            self.reconstruction_loss = self.ae.reconstruction_loss
            self.prediction_loss = self.mlp_loss_scalar * self.mlp.prediction_loss  # scale loss

            # If the decoder is an VAE, incorporate the KL loss. Don't for a regular AE
            if self.variational:
                self.kl_loss = self.ae.beta * self.ae.kl_loss  # scale KL loss with the Beta scalar
                self.total_loss = self.reconstruction_loss + self.kl_loss + self.prediction_loss
            else:
                self.total_loss = self.reconstruction_loss + self.prediction_loss

        self.loss = torch.mean(self.total_loss)

        return sequence_probs, logprobs_N_K_C, z, self.loss

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

        all_token_probs_N_S_C = []
        all_y_logprobs_N_K_C = []
        all_reconstruction_losses = []
        all_pretrained_reconstruction_losses = []
        all_ood_scores = []
        all_kl_losses = []
        all_prediction_losses = []
        all_total_losses = []
        all_ys = []
        all_smiles = []

        self.eval()
        with torch.no_grad():
            for x in val_loader:
                x, y = batch_management(x, self.device)

                # reconvert the encoding to smiles and save them. This is inefficient, but due to on the go smiles
                # augmentation it is impossible to get this info from the dataloader directly
                all_smiles.extend(encoding_to_smiles(x, strip=True))

                # predict
                token_probs_N_S_C, y_logprobs_N_K_C, z, loss = self(x, y)

                all_token_probs_N_S_C.append(token_probs_N_S_C)
                all_y_logprobs_N_K_C.append(y_logprobs_N_K_C)

                if y is not None:
                    all_reconstruction_losses.append(self.reconstruction_loss)
                    if self.variational:
                        all_kl_losses.append(self.kl_loss)
                    all_prediction_losses.append(self.prediction_loss)
                    all_total_losses.append(self.total_loss)
                    all_ys.append(y)

                # if there's a pretrained model loaded, use it to debias the reconstruction loss
                if self.pretrained_ae is not None:
                    ood_score_pt = self.pretrained_decoder.reconstruction_loss
                    all_pretrained_reconstruction_losses.append(ood_score_pt)

                    ood_score = ood_score - ood_score_pt
                    all_ood_scores.append(ood_score)

            all_token_probs_N_S_C = torch.cat(all_token_probs_N_S_C, 0)
            all_y_logprobs_N_K_C = torch.cat(all_y_logprobs_N_K_C, 0)
            reconstruction_loss = torch.cat(all_reconstruction_losses) if len(all_reconstruction_losses) > 0 else None
            pretrained_reconstruction_loss = torch.cat(all_pretrained_reconstruction_losses) if len(all_pretrained_reconstruction_losses) > 0 else None
            ood_score = torch.cat(all_ood_scores) if len(all_ood_scores) > 0 else None
            kl_loss = torch.cat(all_kl_losses) if len(all_kl_losses) > 0 else None
            prediction_loss = torch.cat(all_prediction_losses) if len(all_prediction_losses) > 0 else None
            total_loss = torch.cat(all_total_losses) if len(all_total_losses) > 0 else None
            all_ys = torch.cat(all_ys) if len(all_ys) > 0 else None

            output = {"token_probs_N_S_C": all_token_probs_N_S_C,
                      "y_logprobs_N_K_C": all_y_logprobs_N_K_C,
                      "reconstruction_loss": reconstruction_loss,
                      "pretrained_reconstruction_loss": pretrained_reconstruction_loss,
                      "ood_score": ood_score,
                      "kl_loss": kl_loss,
                      "prediction_loss": prediction_loss,
                      "total_loss": total_loss,
                      "y": all_ys,
                      "smiles": all_smiles}

        return output

    def get_z(self, dataset: MoleculeDataset, batch_size: int = 256) -> (Tensor, list):
        """ Get the latent representation :math:`z` of molecules

        :param dataset: MoleculeDataset that returns a batch of integer encoded molecules :math:`(N, C)`
        :param batch_size: number of samples in a batch
        :return: latent vectors :math:`(N, H)`, where hidden is the VAE compression dimension
        """

        val_loader = get_val_loader(self.config, dataset, batch_size)

        all_z = []
        all_smiles = []

        self.eval()
        with torch.no_grad():
            for x in val_loader:
                x, y = batch_management(x, self.device)
                all_smiles.extend(encoding_to_smiles(x, strip=True))
                sequence_probs, y_logprobs_N_K_C, z, molecule_reconstruction_loss, loss = self(x, y)
                all_z.append(z)

        return torch.cat(all_z), all_smiles


class RfEnsemble:
    """ Ensemble of RFs"""
    def __init__(self, config, **kwargs) -> None:
        super(RfEnsemble, self).__init__()
        self.config = config
        self.ensemble_size = self.config.ensemble_size
        self.seed = 0
        self.seeds = np.random.default_rng(seed=self.seed).integers(0, 1000, self.ensemble_size)
        self.model_hypers = filter_params(RandomForestClassifier.__init__, self.config.hyperparameters)
        class_weight = "balanced" if self.config.balance_classes else None
        self.models = {i: RandomForestClassifier(random_state=s, class_weight=class_weight, **self.model_hypers)
                       for i, s in enumerate(self.seeds)}

    def train(self, x, y, **kwargs) -> None:
        for i, m in self.models.items():
            m.fit(x, y)

    def predict(self, x, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        # logits_N_K_C = torch.stack([m.predict(dataloader) for m in self.models.values()], 1)
        eps = 1e-10  # we need to add this, so we don't get divide by zero errors in our log function

        y_hats = []
        for m in self.models.values():

            y_hat = torch.tensor(m.predict_proba(x) + eps)
            if y_hat.shape[1] == 1:  # if only one class if predicted with the RF model, add a column of zeros
                y_hat = torch.cat((y_hat, torch.zeros((y_hat.shape[0], 1))), dim=1)
            y_hats.append(y_hat)

        logits_N_K_C = torch.stack(y_hats, 1)

        logits_N_K_C = torch.log(logits_N_K_C)

        return logits_N_K_C

    def __getitem__(self, item):
        return self.models[item]

    def __repr__(self) -> str:
        return f"Ensemble of {self.ensemble_size} RF Classifiers"
