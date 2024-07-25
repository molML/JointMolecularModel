
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from jcm.utils import get_val_loader, batch_management, filter_params
from jcm.modules.rnn import AutoregressiveRNN, init_start_tokens, DecoderRNN
from jcm.modules.base import BaseModule
from jcm.modules.cnn import CnnEncoder
from jcm.modules.mlp import Ensemble
from jcm.modules.variational import VariationalEncoder
from jcm.datasets import MoleculeDataset
from jcm.modules.rnn import init_rnn_hidden
from cheminformatics.encoding import encoding_to_smiles


class DeNovoRNN(AutoregressiveRNN, BaseModule):
    # SMILES -> RNN -> SMILES

    def __init__(self, config, **kwargs):
        self.config = config
        super(DeNovoRNN, self).__init__(**self.config.hyperparameters)

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
        all_lossses = []
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
            all_lossses.append(loss)

        all_probs = torch.cat(all_probs, 0)
        all_sample_losses = torch.cat(all_sample_losses, 0)
        all_lossses = torch.mean(torch.stack(all_lossses))

        return all_probs, all_sample_losses, all_lossses, all_smiles


class VAE(BaseModule):
    # SMILES -> CNN -> variational -> RNN -> SMILES
    def __init__(self, config, **kwargs):
        super(VAE, self).__init__()

        self.config = config
        self.device = config.hyperparameters['device']
        self.register_buffer('beta', torch.tensor(config.hyperparameters['beta']))

        self.cnn = CnnEncoder(**config.hyperparameters)
        self.variational_layer = VariationalEncoder(var_input_dim=self.cnn.out_dim, **config.hyperparameters)
        self.rnn = DecoderRNN(**self.config.hyperparameters)

    def forward(self, x: Tensor, y: Tensor = None) -> (Tensor, Tensor, Tensor, Tensor):
        """ Reconstruct a batch of molecule

        :param x: :math:`(N, C)`, batch of integer encoded molecules
        :param y: does nothing, here for compatibility sake
        :return: sequence_probs, z, molecule_loss, loss
        """

        # Embed the integer encoded molecules with the same embedding layer that is used later in the rnn
        # We transpose it from (batch size x sequence length x embedding) to (batch size x embedding x sequence length)
        # so the embedding is the channel instead of the sequence length
        embedding = self.rnn.embedding_layer(x).transpose(1, 2)

        # Encode the molecule into a latent vector z
        z = self.variational_layer(self.cnn(embedding))

        # Decode z back into a molecule
        sequence_probs, molecule_loss, loss = self.rnn(z, x)

        # Add the KL-divergence loss from the variational layer
        loss_kl = self.variational_layer.kl / x.shape[0]
        loss = loss + self.beta * loss_kl

        return sequence_probs, z, molecule_loss, loss

    @BaseModule().inference
    def generate(self, z: Tensor = None, seq_length: int = 101, n: int = 1, batch_size: int = 256) -> Tensor:
        """ Generate molecules from either a tensor of latent representations or random tensors

        :param z: Tensor (N, Z)
        :param seq_length: number of tokens to generate
        :param n: number of molecules to generate. Only applies when z = None, else takes the first dim of z as n
        :param batch_size: size of the batches
        :return: Tensor (N, S, C)
        """

        if z is None:
            chunks = [batch_size] * (n // batch_size) + ([n % batch_size] if n % batch_size else [])
            all_probs = []
            for chunk in chunks:
                z_ = torch.rand(chunk, self.rnn.z_size)
                all_probs.append(self.rnn.generate_from_z(z_, seq_len=seq_length+1))
        else:
            n = z.size(0)
            chunks = [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]
            all_probs = []
            for chunk in chunks:
                z_ = z[chunk]
                all_probs.append(self.rnn.generate_from_z(z_, seq_len=seq_length+1))

        return torch.cat(all_probs)

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
        all_lossses = []


        for x in val_loader:
            x, y = batch_management(x, self.device)

            # reconvert the encoding to smiles and save them. This is inefficient, but due to on the go smiles
            # augmentation it is impossible to get this info from the dataloader directly
            all_smiles.extend(encoding_to_smiles(x, strip=True))

            # predict
            sequence_probs, z, molecule_loss, loss = self(x)

            all_probs.append(sequence_probs)
            all_molecule_losses.append(molecule_loss)
            all_lossses.append(loss)

        all_probs = torch.cat(all_probs, 0)
        all_molecule_losses = torch.cat(all_molecule_losses, 0)
        all_lossses = torch.mean(torch.stack(all_lossses))

        return all_probs, all_molecule_losses, all_lossses, all_smiles

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

            embedding = self.rnn.embedding_layer(x).transpose(1, 2)
            # Encode the molecule into a latent vector z
            z = self.variational_layer(self.cnn(embedding))
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

        # Embed the integer encoded molecules with the same embedding layer that is used later in the rnn
        # We transpose it from (batch size x sequence length x embedding) to (batch size x embedding x sequence length)
        # so the embedding is the channel instead of the sequence length
        embedding = self.embedding_layer(x).transpose(1, 2)

        # Encode the molecule into a latent vector z
        z = self.variational_layer(self.cnn(embedding))

        # Predict a property from this embedding
        y_logprobs_N_K_C, molecule_loss, loss = self.mlp(z, y)

        # Add the KL-divergence loss from the variational layer
        if loss is not None:
            loss_kl = self.variational_layer.kl / x.shape[0]
            loss = loss + self.beta * loss_kl

        return y_logprobs_N_K_C, z, loss

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
        :return: class log probabilities :math:`(N, K, C)`, where K is ensemble size, loss, and target labels :math:`(N)`
        """

        val_loader = get_val_loader(self.config, dataset, batch_size, sample)

        all_y_logprobs_N_K_C = []
        all_ys = []
        all_losses = []

        for x in val_loader:
            x, y = batch_management(x, self.device)

            # predict
            y_logprobs_N_K_C, z, loss = self(x, y)

            all_y_logprobs_N_K_C.append(y_logprobs_N_K_C)
            if y is not None:
                all_losses.append(loss)
                all_ys.append(y)

        all_y_logprobs_N_K_C = torch.cat(all_y_logprobs_N_K_C, 0)
        all_ys = torch.cat(all_ys) if len(all_ys) > 0 else None
        all_losses = torch.mean(torch.cat(all_losses)) if len(all_losses) > 0 else None

        return all_y_logprobs_N_K_C, all_losses, all_ys

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
            y_logprobs_N_K_C, z, loss = self(x)
            all_z.append(z)

        return torch.cat(all_z), all_smiles


class MLP(Ensemble, BaseModule):
    # ECFP -> MLP -> yhat

    def __init__(self, config, **kwargs):
        self.config = config
        super(MLP, self).__init__(**self.config.hyperparameters)

    @BaseModule().inference
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
        all_losses = []

        for x in val_loader:
            x, y = batch_management(x, self.device)

            # predict
            y_logprobs_N_K_C, _, loss = self(x, y)

            all_y_logprobs_N_K_C.append(y_logprobs_N_K_C)
            if y is not None:
                all_losses.append(loss)
                all_ys.append(y)

        all_y_logprobs_N_K_C = torch.cat(all_y_logprobs_N_K_C, 0)
        all_ys = torch.cat(all_ys) if len(all_ys) > 0 else None
        all_losses = torch.mean(torch.cat(all_losses)) if len(all_losses) > 0 else None

        return all_y_logprobs_N_K_C, all_losses, all_ys


class JointChemicalModel(BaseModule):
    # SMILES -> CNN -> variational -> rnn -> SMILES
    #                            |
    #                           MLP -> property
    def __init__(self, config, **kwargs):
        self.config = config
        self.device = self.config.hyperparameters['device']
        super(JointChemicalModel, self).__init__()

        self.vae = VAE(config)
        self.mlp = MLP(config)
        self.register_buffer('mlp_loss_scalar', torch.tensor(config.hyperparameters['mlp_loss_scalar']))

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
        sequence_probs, z, molecule_reconstruction_loss, vae_loss = self.vae(x)

        # predict property from latent representation
        logprobs_N_K_C, mlp_molecule_loss, mlp_loss = self.mlp(z, y)

        # combine losses, but if y is None, return the loss as None
        if mlp_loss is None:
            loss = None
        else:
            loss = vae_loss + self.mlp_loss_scalar * mlp_loss

        return sequence_probs, logprobs_N_K_C, z, molecule_reconstruction_loss, loss

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

        all_token_probs_N_S_C = []
        all_y_logprobs_N_K_C = []
        all_molecule_reconstruction_losses = []
        all_losses = []
        all_ys = []
        all_smiles = []

        for x in val_loader:
            x, y = batch_management(x, self.device)

            # reconvert the encoding to smiles and save them. This is inefficient, but due to on the go smiles
            # augmentation it is impossible to get this info from the dataloader directly
            all_smiles.extend(encoding_to_smiles(x, strip=True))

            # predict
            token_probs_N_S_C, y_logprobs_N_K_C, z, molecule_reconstruction_loss, loss = self(x, y)

            all_token_probs_N_S_C.append(token_probs_N_S_C)
            all_y_logprobs_N_K_C.append(y_logprobs_N_K_C)
            all_molecule_reconstruction_losses.append(molecule_reconstruction_loss)

            if y is not None:
                all_losses.append(loss)
                all_ys.append(y)

        all_token_probs_N_S_C = torch.cat(all_token_probs_N_S_C, 0)
        all_y_logprobs_N_K_C = torch.cat(all_y_logprobs_N_K_C, 0)
        all_molecule_reconstruction_losses = torch.cat(all_molecule_reconstruction_losses)
        all_ys = torch.cat(all_ys) if len(all_ys) > 0 else None
        all_losses = torch.mean(torch.cat(all_losses)) if len(all_losses) > 0 else None

        return all_token_probs_N_S_C, all_y_logprobs_N_K_C, all_molecule_reconstruction_losses, all_losses, all_ys, \
            all_smiles

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
        self.models = {i: RandomForestClassifier(random_state=s, class_weight="balanced", **self.model_hypers)
                       for i, s in enumerate(self.seeds)}

    def train(self, x, y, **kwargs) -> None:
        for i, m in self.models.items():
            m.fit(x, y)

    def predict(self, x, **kwargs) -> Tensor:
        """ logits_N_K_C = [N, num_inference_samples, num_classes] """
        # logits_N_K_C = torch.stack([m.predict(dataloader) for m in self.models.values()], 1)
        eps = 1e-10  # we need to add this so we don't get divide by zero errors in our log function

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
