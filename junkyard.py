


import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data.dataloader import DataLoader
from cheminformatics.descriptors import one_hot_encode, encoding_to_smiles, mols_to_ecfp
from jcm.utils import BCE_per_sample, single_batchitem_fix, calc_l_out
from jcm.config import Config
from rdkit import Chem
from constants import VOCAB


class CnnEncoder(nn.Module):
    """ Encode a one-hot encoded SMILES string with a CNN. Uses Max Pooling and flattens conv layer at the end

    :param channels: vocab size (default=36)
    :param seq_length: sequence length of SMILES strings (default=102)
    :param out_hidden: dimension of the CNN token embedding size (default=256)
    :param kernel_size: CNN kernel_size (default=8)
    :param n_layers: number of layers in the CNN (default=3)
    :param stride: stride (default=1)
    """

    def __init__(self, channels: int = 36, seq_length: int = 102, out_hidden: int = 256, kernel_size: int = 8,
                 stride: int = 1, n_layers: int = 3, **kwargs):
        super().__init__()
        self.n_layers = n_layers

        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
        if n_layers == 1:
            self.cnn0 = nn.Conv1d(channels, out_hidden, kernel_size=kernel_size, stride=stride)
            self.l_out = calc_l_out(seq_length, self.cnn0, self.pool)
        if n_layers == 2:
            self.cnn0 = nn.Conv1d(channels, 128, kernel_size=kernel_size, stride=stride)
            self.cnn1 = nn.Conv1d(128, out_hidden, kernel_size=kernel_size, stride=stride)
            self.l_out = calc_l_out(seq_length, self.cnn0, self.pool, self.cnn1, self.pool)
        if n_layers == 3:
            self.cnn0 = nn.Conv1d(channels, 64, kernel_size=kernel_size, stride=stride)
            self.cnn1 = nn.Conv1d(64, 128, kernel_size=kernel_size, stride=stride)
            self.cnn2 = nn.Conv1d(128, out_hidden, kernel_size=kernel_size, stride=stride)
            self.l_out = calc_l_out(seq_length, self.cnn0, self.pool, self.cnn1, self.pool, self.cnn2, self.pool)

        self.out_dim = int(out_hidden * self.l_out)

    def forward(self, x):
        x = F.relu(self.cnn0(x))
        x = self.pool(x)

        if self.n_layers == 2:
            x = F.relu(self.cnn1(x))
            x = self.pool(x)

        if self.n_layers == 3:
            x = F.relu(self.cnn1(x))
            x = self.pool(x)
            x = F.relu(self.cnn2(x))
            x = self.pool(x)

        # flatten
        x = x.view(x.size(0), -1)

        return x


class LSTMDecoder(nn.Module):
    """ A conditioned LSTM that decodes a batch of latent vectors (batch_size, embedding_size) into
    SMILES strings logits (batch_size, vocab_size, sequence_length). Supports trainer forcing if provided with the
    true X sequence.

    :param hidden_size: size of the hidden layers in the LSTM (default=64)
    :param vocabulary_size: number of tokens in the vocab (default=36)
    :param sequence_length: length of the SMILES strings (default=102)
    :param teacher_forcing_prob: the probability of teacher forcing being used when generating a token (default=0.5)
    :param device: device (can be 'cpu' or 'cuda')
    """

    def __init__(self, hidden_size: int, vocabulary_size: int, sequence_length: int, device: str,
                 teacher_forcing_prob: float = 0.75, learnable_cell_state: bool = True, **kwargs):
        super(LSTMDecoder, self).__init__()
        self.device = device
        self.teacher_forcing_prob = teacher_forcing_prob
        self.learnable_cell_state = learnable_cell_state
        self.hidden_size = hidden_size
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length

        self.lstm = nn.LSTMCell(vocabulary_size, hidden_size)
        self.cell_state_from_z = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, z, x=None, sequence_length: int = None) -> Tensor:

        batch_size = z.size(0)
        sequence_length = self.sequence_length if sequence_length is None else sequence_length

        # initiate the hidden state and the cell state (fresh)
        hidden_state = z  # should be z

        if self.learnable_cell_state:
            cell_state = F.relu(self.cell_state_from_z(z))
        else:
            cell_state = torch.zeros(batch_size, self.hidden_size, device=self.device)  # zero or maybe relu(lin(z)) ?

        # initiate the start token
        token = torch.zeros((batch_size, self.vocabulary_size), device=self.device)
        token[:, 0] = 1.

        output = []
        for i in range(sequence_length - 1):

            # random change of teacher forcing if x is provided
            if x is not None:
                if torch.rand(1) < self.teacher_forcing_prob:
                    token = x[:, i]  # x should be a float

            hidden_state, cell_state = self.lstm(token, (hidden_state, cell_state))

            # save logits of the predicted token so we can calculate the loss
            token_logits = F.relu(self.fc(hidden_state))
            output.append(token_logits)

            # Get the token for the next prediction (in one-hot-encoded format)
            token = one_hot_encode(F.softmax(token_logits, -1).argmax(-1)).float()

        # stack all outputs on top of each other to form the full sequence
        output = torch.stack(output, 1)

        return output


class LstmVAE(nn.Module):
    """ A  LstmVAE, where the latent space z is used as an input for the MLP.

    :param vocab_size: size of the vocabulary (default=36)
    :param latent_dim: dimensions of the latent layer (default=64)
    :param hidden_dim: dimensions of the hidden layer(s) of the CNN encoder (default=256)
    :param kernel_size: CNN kernel size (default=8)
    :param beta: scales the KL loss (default=0.001)
    :param seq_length: length of the SMILES sequences (default=102)
    :param variational_scale: The scale of the Gaussian of the encoder (default=1)
    :param teacher_forcing_prob: the probability of teacher forcing being used when generating a token (default=0.5)
    :param device: device (default=None, can be 'cuda' or 'cpu')
    :param n_layers: number of layers in the CNN (default=3)
    :param kwargs: Just here for compatability
    """

    def __init__(self, vocab_size: int = 36, latent_dim: int = 128, hidden_dim_cnn: int = 256,
                 hidden_dim_lstm: int = 256, kernel_size: int = 8, beta: float = 0.001, seq_length: int = 102,
                 variational_scale: float = 1, device: str = None, teacher_forcing_prob: float = 0.75,
                 n_layers_cnn: int = 3, start_token_weight: float = 10, **kwargs):
        super(LstmVAE, self).__init__()
        self.name = 'LstmVAE'
        self.device = device
        self.register_buffer('beta', torch.tensor(beta))

        self.cnn = CnnEncoder(channels=vocab_size, seq_length=seq_length,  out_hidden=hidden_dim_cnn,
                              kernel_size=kernel_size, n_layers=n_layers_cnn)

        self.variational_layer = VariationalEncoder(input_dim=self.cnn.out_dim, latent_dim=latent_dim,
                                                    variational_scale=variational_scale)

        self.z_projection = nn.Linear(latent_dim, hidden_dim_lstm)

        self.decoder = LSTMDecoder(hidden_dim_lstm, vocab_size, seq_length, device=self.device,
                                   teacher_forcing_prob=teacher_forcing_prob)

        self.cw = torch.ones(VOCAB['vocab_size'], device=self.device)
        self.cw[VOCAB['start_idx']] = start_token_weight

    def forward(self, x: Tensor, y: Tensor = None) -> (Tensor, Tensor, Tensor, Tensor):

        # turn indexed encoding into one-hots w. shape N, C, L
        x_oh = one_hot_encode(x.long()).float()

        # Get latent vectors and decode them back into a molecule
        z = self.cnn(x_oh.transpose(1, 2))
        z = self.variational_layer(z)
        z_ = F.relu(self.z_projection(z))
        x_hat = self.decoder(z_, x_oh)

        # compute losses
        sample_likelihood, loss_reconstruction = token_loss(x_hat, x.long(), ignore_index=VOCAB['pad_idx'],
                                                            weight=self.cw)
        loss_kl = self.variational_layer.kl / x.shape[0]  # divide by batch size
        loss = loss_reconstruction + self.beta * loss_kl  # add the reconstruction loss and the scaled KL loss

        return x_hat, z, sample_likelihood, loss

    def predict(self, dataset, batch_size: int = 128) -> Tensor:
        return _predict_lstm_vae(self, dataset, batch_size)


class LstmJVAE(nn.Module):
    """ A joint LstmVAE, where the latent space z is used as an input for the MLP.

    :param vocab_size: size of the vocabulary (default=36)
    :param latent_dim: dimensions of the latent layer (default=64)
    :param hidden_dim_vae: dimensions of the hidden layer(s) of the decoder (default=2048)
    :param kernel_size: CNN kernel size (default=8)
    :param beta: scales the KL loss (default=0.001)
    :param seq_length: length of the SMILES sequences (default=102)
    :param n_layers_mlp: number of MLP layers (including the input layer, not including the output layer, default=2)
    :param hidden_dim_mlp: hidden layer(s) dimension of the MLP (default=2048)
    :param anchored: toggles weight anchoring of the MLP (default=False)
    :param l2_lambda: L2 loss scaling for the anchored MLP loss (default=1e-4)
    :param n_ensemble: number of MLPs in the ensemble (default=10)
    :param output_dim_mlp: dimensions of the output layer of the MLP (default=2)
    :param class_scaling_factor: Scales BCE loss for the '1' class, i.e. a factor of 2 would double the respective loss
        for the '1' class (default=None)
    :param variational_scale: The scale of the Gaussian of the encoder (default=1)
    :param kwargs: just here for compatability reasons.
    """

    def __init__(self, vocab_size: int = 36, latent_dim: int = 64, hidden_dim_vae: int = 256, kernel_size: int = 8,
                 beta: float = 0.001, n_layers_mlp: int = 2, hidden_dim_mlp: int = 2048, anchored: bool = True,
                 seq_length: int = 102,  l2_lambda: float = 1e-4, n_ensemble: int = 10, output_dim_mlp: int = 2,
                 variational_scale: float = 1, device: str = None, mlp_loss_scalar: float = 1, **kwargs) -> None:
        super(LstmJVAE, self).__init__()
        self.name = 'LstmJVAE'
        self.device = device
        self.register_buffer('mlp_loss_scalar', torch.tensor(mlp_loss_scalar))

        self.vae = LstmVAE(vocab_size=vocab_size, latent_dim=latent_dim, hidden_dim=hidden_dim_vae, beta=beta,
                           kernel_size=kernel_size, variational_scale=variational_scale, seq_length=seq_length)

        self.prediction_head = Ensemble(input_dim=latent_dim, hidden_dim=hidden_dim_mlp, n_layers=n_layers_mlp,
                                        anchored=anchored, l2_lambda=l2_lambda, n_ensemble=n_ensemble,
                                        output_dim=output_dim_mlp)

    def forward(self, x: Tensor, y: Tensor = None, **kwargs) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):

        x_hat, z, sample_likelihood, loss = self.vae(x)
        y_logits_N_K_C, loss_mlp_i, loss_mlp = self.prediction_head(z, y)

        if y is not None:
            loss = loss + self.mlp_loss_scalar * loss_mlp

        return y_logits_N_K_C, x_hat, z, sample_likelihood, loss_mlp_i, loss

    def predict(self, dataset, pretrained_vae_path: str = None, batch_size: int = 128) -> Tensor:
        return _predict_lstm_jvae(self, dataset, pretrained_vae_path=pretrained_vae_path, batch_size=batch_size)


class Decoder(nn.Module):
    """ A decoder that reconstructs a binary vector from a latent representation

    :param input_dim: dimensions of the input layer (default=2048)
    :param hidden_dim: dimensions of the hidden layer(s) (default=2048)
    :param out_dim: dimensions of the output layer (default=2)
    :param n_layers: number of layers (including the input layer, not including the output layer, default=2)
    """
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 2048, out_dim: int = 2048, n_layers: int = 1,
                 **kwargs) -> None:
        super(Decoder, self).__init__()
        self.name = 'Decoder'
        self.fc = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.fc:
            lin.reset_parameters()
        self.out.reset_parameters()

    def forward(self, z: Tensor) -> Tensor:
        for lin in self.fc:
            z = F.relu(lin(z))
        z = torch.sigmoid(self.out(z))

        return z


class VariationalEncoder(nn.Module):
    """ A simple variational encoder. Takes a batch of vectors and compresses the input space to a smaller variational
    latent.

    :param input_dim: dimensions of the input layer (default=2048)
    :param latent_dim: dimensions of the latent/output layer (default=2048)
    :param variational_scale: The scale of the Gaussian of the encoder (default=1)
    """
    def __init__(self, input_dim: int = 2048, latent_dim: int = 128, variational_scale: float = 1, **kwargs):
        super(VariationalEncoder, self).__init__()
        self.name = 'VariationalEncoder'

        self.lin0_x = nn.Linear(input_dim, latent_dim)
        self.lin0_mu = nn.Linear(latent_dim, latent_dim)
        self.lin0_sigma = nn.Linear(latent_dim, latent_dim)

        self.N = torch.distributions.Normal(0, variational_scale)
        self.N.loc = self.N.loc
        self.N.scale = self.N.scale
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.lin0_x(x))
        mu = self.lin0_mu(x)
        sigma = torch.exp(self.lin0_sigma(x))

        # reparameterization trick
        z = mu + sigma * self.N.sample(mu.shape).to('cuda' if torch.cuda.is_available() else 'cpu')  # TODO fix this
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()

        return z


@torch.no_grad()
def _predict_vae(model, dataset, batch_size: int = 128) -> (Tensor, Tensor, Tensor):
    """ Get predictions from a dataloader

    :param model: torch module (e.g. MLP or Ensemble)
    :param dataset: dataset of the data to predict; jcm.datasets.MoleculeDataset
    :param batch_size: prediction batch size (default=128)

    :return: logits_N_K_C, vae latents, vae likelihoods
    """
    val_loader = DataLoader(dataset, sampler=None, shuffle=False, pin_memory=True, batch_size=batch_size,
                            collate_fn=single_batchitem_fix)
    y_hats = []
    zs = []
    sample_likelihoods = []

    model.eval()
    for x, y in val_loader:
        # move to device
        x.to(model.device)

        # predict
        y_hat, z, sample_likelihood, loss = model(x)

        y_hats.append(y_hat)
        zs.append(z)
        sample_likelihoods.append(sample_likelihood)

    model.train()

    y_hats = torch.cat(y_hats, 0)
    zs = torch.cat(zs)
    sample_likelihoods = torch.cat(sample_likelihoods, 0)

    return y_hats, zs, sample_likelihoods


@torch.no_grad()
def _predict_jvae(model, dataset, pretrained_vae_path: str = None, batch_size: int = 128) -> (Tensor, Tensor, Tensor):
    """ Get predictions from a dataloader

    :param model: torch module (e.g. MLP or Ensemble)
    :param dataset: dataset of the data to predict; jcm.datasets.MoleculeDataset
    :param pretrained_vae_path: path of the pretrained EcfpVAE, used to normalize likelihoods if supplied (default=None)
    :param batch_size: prediction batch size (default=128)

    :return: logits_N_K_C, x_reconstructions, vae latents, vae likelihoods
    """

    if pretrained_vae_path is not None:
        config = Config()
        # config.set_hyperparameters(**VAE_PRETRAIN_HYPERPARAMETERS)    TODO FIX THIS

        pre_trained_vae = LstmVAE(**config.hyperparameters)
        pre_trained_vae.load_state_dict(torch.load(pretrained_vae_path))
        pre_trained_vae.eval()

    val_loader = DataLoader(dataset, sampler=None, shuffle=False, pin_memory=True, batch_size=batch_size,
                            collate_fn=single_batchitem_fix)

    y_hats = []
    x_hats = []
    zs = []
    sample_likelihoods = []
    pt_sample_likelihoods = []

    model.eval()
    for x, y in val_loader:
        # move to device
        x.to(model.device)

        # predict
        y_hat, x_hat, z, sample_likelihood, loss = model(x)

        if pretrained_vae_path is not None:
            pt_x_hat, pt_z, pt_sample_likelihood, pt_loss = pre_trained_vae(x)

        y_hats.append(y_hat)
        zs.append(z)
        x_hats.append(x_hat)
        sample_likelihoods.append(sample_likelihood)
        pt_sample_likelihoods.append(pt_sample_likelihood)

    model.train()

    y_hats = torch.cat(y_hats, 0)
    x_hats = torch.cat(x_hats)
    zs = torch.cat(zs)
    sample_likelihoods = torch.cat(sample_likelihoods, 0)

    # normalize X likelihoods using the pretrained likelihoods
    if pretrained_vae_path is not None:
        pt_sample_likelihoods = torch.cat(pt_sample_likelihoods, 0)
        sample_likelihoods = sample_likelihoods - pt_sample_likelihoods

    return y_hats, x_hats, zs, sample_likelihoods


@torch.no_grad()
def _predict_lstm_vae(model, dataset, batch_size: int = 128) -> (Tensor, Tensor, Tensor):
    """ Get predictions from a dataloader

    :param model: torch module (e.g. MLP or Ensemble)
    :param dataset: dataset of the data to predict; jcm.datasets.MoleculeDataset
    :param batch_size: prediction batch size (default=128)

    :return: logits_N_K_C, vae latents, vae likelihoods
    """
    val_loader = DataLoader(dataset, sampler=None, shuffle=False, pin_memory=True, batch_size=batch_size,
                            collate_fn=single_batchitem_fix)
    x_hats = []
    zs = []
    sample_likelihoods = []

    model.eval()
    for x in val_loader:
        # move to device
        x = x.to(model.device)

        # predict
        x_hat, z, sample_likelihood, loss = model(x)

        x_hats.append(x_hat.cpu())
        zs.append(z.cpu())
        sample_likelihoods.append(sample_likelihood.cpu())

    model.train()

    y_hats = torch.cat(x_hats, 0)
    zs = torch.cat(zs)
    sample_likelihoods = torch.cat(sample_likelihoods, 0)

    return y_hats, zs, sample_likelihoods


@torch.no_grad()
def _predict_lstm_jvae(model, dataset, pretrained_vae_path: str = None, batch_size: int = 128) -> \
        (Tensor, Tensor, Tensor):
    """ Get predictions from a dataloader

    :param model: torch module (e.g. MLP or Ensemble)
    :param dataset: dataset of the data to predict; jcm.datasets.MoleculeDataset
    :param pretrained_vae_path: path of the pretrained EcfpVAE, used to normalize likelihoods if supplied (default=None)
    :param batch_size: prediction batch size (default=128)

    :return: logits_N_K_C, x_reconstructions, vae latents, vae likelihoods
    """

    if pretrained_vae_path is not None:
        config = Config()
        # config.set_hyperparameters(**VAE_PRETRAIN_HYPERPARAMETERS)  TODO FIX THIS

        pre_trained_vae = LstmVAE(**config.hyperparameters)
        pre_trained_vae.load_state_dict(torch.load(pretrained_vae_path))
        pre_trained_vae.eval()

    val_loader = DataLoader(dataset, sampler=None, shuffle=False, pin_memory=True, batch_size=batch_size,
                            collate_fn=single_batchitem_fix)

    y_hats = []
    x_hats = []
    zs = []
    sample_likelihoods = []
    pt_sample_likelihoods = []

    model.eval()
    for x, y in val_loader:
        # move to device
        x.to(model.device)

        # predict
        y_hat, x_hat, z, sample_likelihood, loss = model(x.long())

        if pretrained_vae_path is not None:
            pt_x_hat, pt_z, pt_sample_likelihood, pt_loss = pre_trained_vae(x.long())

        y_hats.append(y_hat)
        zs.append(z)
        x_hats.append(x_hat)
        sample_likelihoods.append(sample_likelihood)
        pt_sample_likelihoods.append(pt_sample_likelihood)

    model.train()

    y_hats = torch.cat(y_hats, 0)
    x_hats = torch.cat(x_hats)
    zs = torch.cat(zs)
    sample_likelihoods = torch.cat(sample_likelihoods, 0)

    # normalize X likelihoods using the pretrained likelihoods
    if pretrained_vae_path is not None:
        pt_sample_likelihoods = torch.cat(pt_sample_likelihoods, 0)
        sample_likelihoods = sample_likelihoods - pt_sample_likelihoods

    return y_hats, x_hats, zs, sample_likelihoods


def token_loss(logits: Tensor, target: Tensor, ignore_index: int = -1, weight: Tensor = None):
    """

    :param logits: model output in the shape of (batch_size, sequence_length, vocab)
    :param target: target token indices in the shape of (batch_size, sequence_length)
    :param ignore_index: token to ignore
    :param weight: tensor with shape C that weights all tokens (default = None, 1 = standard weight)
    :return:
    """
    loss_func = nn.NLLLoss(reduction='none', ignore_index=ignore_index, weight=weight)

    x_hat = F.log_softmax(logits, dim=-1)

    loss = loss_func(x_hat.transpose(2, 1), target[:, 1:])
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data.dataloader import DataLoader
from cheminformatics.descriptors import one_hot_encode, encoding_to_smiles, mols_to_ecfp
from jcm.utils import BCE_per_sample, single_batchitem_fix, calc_l_out
from jcm.config import Config
from rdkit import Chem
from constants import VOCAB


class CnnEncoder(nn.Module):
    """ Encode a one-hot encoded SMILES string with a CNN. Uses Max Pooling and flattens conv layer at the end

    :param channels: vocab size (default=36)
    :param seq_length: sequence length of SMILES strings (default=102)
    :param out_hidden: dimension of the CNN token embedding size (default=256)
    :param kernel_size: CNN kernel_size (default=8)
    :param n_layers: number of layers in the CNN (default=3)
    :param stride: stride (default=1)
    """

    def __init__(self, channels: int = 36, seq_length: int = 102, out_hidden: int = 256, kernel_size: int = 8,
                 stride: int = 1, n_layers: int = 3, **kwargs):
        super().__init__()
        self.n_layers = n_layers

        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
        if n_layers == 1:
            self.cnn0 = nn.Conv1d(channels, out_hidden, kernel_size=kernel_size, stride=stride)
            self.l_out = calc_l_out(seq_length, self.cnn0, self.pool)
        if n_layers == 2:
            self.cnn0 = nn.Conv1d(channels, 128, kernel_size=kernel_size, stride=stride)
            self.cnn1 = nn.Conv1d(128, out_hidden, kernel_size=kernel_size, stride=stride)
            self.l_out = calc_l_out(seq_length, self.cnn0, self.pool, self.cnn1, self.pool)
        if n_layers == 3:
            self.cnn0 = nn.Conv1d(channels, 64, kernel_size=kernel_size, stride=stride)
            self.cnn1 = nn.Conv1d(64, 128, kernel_size=kernel_size, stride=stride)
            self.cnn2 = nn.Conv1d(128, out_hidden, kernel_size=kernel_size, stride=stride)
            self.l_out = calc_l_out(seq_length, self.cnn0, self.pool, self.cnn1, self.pool, self.cnn2, self.pool)

        self.out_dim = int(out_hidden * self.l_out)

    def forward(self, x):
        x = F.relu(self.cnn0(x))
        x = self.pool(x)

        if self.n_layers == 2:
            x = F.relu(self.cnn1(x))
            x = self.pool(x)

        if self.n_layers == 3:
            x = F.relu(self.cnn1(x))
            x = self.pool(x)
            x = F.relu(self.cnn2(x))
            x = self.pool(x)

        # flatten
        x = x.view(x.size(0), -1)

        return x


class LSTMDecoder(nn.Module):
    """ A conditioned LSTM that decodes a batch of latent vectors (batch_size, embedding_size) into
    SMILES strings logits (batch_size, vocab_size, sequence_length). Supports trainer forcing if provided with the
    true X sequence.

    :param hidden_size: size of the hidden layers in the LSTM (default=64)
    :param vocabulary_size: number of tokens in the vocab (default=36)
    :param sequence_length: length of the SMILES strings (default=102)
    :param teacher_forcing_prob: the probability of teacher forcing being used when generating a token (default=0.5)
    :param device: device (can be 'cpu' or 'cuda')
    """

    def __init__(self, hidden_size: int, vocabulary_size: int, sequence_length: int, device: str,
                 teacher_forcing_prob: float = 0.75, learnable_cell_state: bool = True, **kwargs):
        super(LSTMDecoder, self).__init__()
        self.device = device
        self.teacher_forcing_prob = teacher_forcing_prob
        self.learnable_cell_state = learnable_cell_state
        self.hidden_size = hidden_size
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length

        self.lstm = nn.LSTMCell(vocabulary_size, hidden_size)
        self.cell_state_from_z = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, z, x=None, sequence_length: int = None) -> Tensor:

        batch_size = z.size(0)
        sequence_length = self.sequence_length if sequence_length is None else sequence_length

        # initiate the hidden state and the cell state (fresh)
        hidden_state = z  # should be z

        if self.learnable_cell_state:
            cell_state = F.relu(self.cell_state_from_z(z))
        else:
            cell_state = torch.zeros(batch_size, self.hidden_size, device=self.device)  # zero or maybe relu(lin(z)) ?

        # initiate the start token
        token = torch.zeros((batch_size, self.vocabulary_size), device=self.device)
        token[:, 0] = 1.

        output = []
        for i in range(sequence_length - 1):

            # random change of teacher forcing if x is provided
            if x is not None:
                if torch.rand(1) < self.teacher_forcing_prob:
                    token = x[:, i]  # x should be a float

            hidden_state, cell_state = self.lstm(token, (hidden_state, cell_state))

            # save logits of the predicted token so we can calculate the loss
            token_logits = F.relu(self.fc(hidden_state))
            output.append(token_logits)

            # Get the token for the next prediction (in one-hot-encoded format)
            token = one_hot_encode(F.softmax(token_logits, -1).argmax(-1)).float()

        # stack all outputs on top of each other to form the full sequence
        output = torch.stack(output, 1)

        return output


class LstmVAE(nn.Module):
    """ A  LstmVAE, where the latent space z is used as an input for the MLP.

    :param vocab_size: size of the vocabulary (default=36)
    :param latent_dim: dimensions of the latent layer (default=64)
    :param hidden_dim: dimensions of the hidden layer(s) of the CNN encoder (default=256)
    :param kernel_size: CNN kernel size (default=8)
    :param beta: scales the KL loss (default=0.001)
    :param seq_length: length of the SMILES sequences (default=102)
    :param variational_scale: The scale of the Gaussian of the encoder (default=1)
    :param teacher_forcing_prob: the probability of teacher forcing being used when generating a token (default=0.5)
    :param device: device (default=None, can be 'cuda' or 'cpu')
    :param n_layers: number of layers in the CNN (default=3)
    :param kwargs: Just here for compatability
    """

    def __init__(self, vocab_size: int = 36, latent_dim: int = 128, hidden_dim_cnn: int = 256,
                 hidden_dim_lstm: int = 256, kernel_size: int = 8, beta: float = 0.001, seq_length: int = 102,
                 variational_scale: float = 1, device: str = None, teacher_forcing_prob: float = 0.75,
                 n_layers_cnn: int = 3, start_token_weight: float = 10, **kwargs):
        super(LstmVAE, self).__init__()
        self.name = 'LstmVAE'
        self.device = device
        self.register_buffer('beta', torch.tensor(beta))

        self.cnn = CnnEncoder(channels=vocab_size, seq_length=seq_length,  out_hidden=hidden_dim_cnn,
                              kernel_size=kernel_size, n_layers=n_layers_cnn)

        self.variational_layer = VariationalEncoder(input_dim=self.cnn.out_dim, latent_dim=latent_dim,
                                                    variational_scale=variational_scale)

        self.z_projection = nn.Linear(latent_dim, hidden_dim_lstm)

        self.decoder = LSTMDecoder(hidden_dim_lstm, vocab_size, seq_length, device=self.device,
                                   teacher_forcing_prob=teacher_forcing_prob)

        self.cw = torch.ones(VOCAB['vocab_size'], device=self.device)
        self.cw[VOCAB['start_idx']] = start_token_weight

    def forward(self, x: Tensor, y: Tensor = None) -> (Tensor, Tensor, Tensor, Tensor):

        # turn indexed encoding into one-hots w. shape N, C, L
        x_oh = one_hot_encode(x.long()).float()

        # Get latent vectors and decode them back into a molecule
        z = self.cnn(x_oh.transpose(1, 2))
        z = self.variational_layer(z)
        z_ = F.relu(self.z_projection(z))
        x_hat = self.decoder(z_, x_oh)

        # compute losses
        sample_likelihood, loss_reconstruction = token_loss(x_hat, x.long(), ignore_index=VOCAB['pad_idx'],
                                                            weight=self.cw)
        loss_kl = self.variational_layer.kl / x.shape[0]  # divide by batch size
        loss = loss_reconstruction + self.beta * loss_kl  # add the reconstruction loss and the scaled KL loss

        return x_hat, z, sample_likelihood, loss

    def predict(self, dataset, batch_size: int = 128) -> Tensor:
        return _predict_lstm_vae(self, dataset, batch_size)


class LstmJVAE(nn.Module):
    """ A joint LstmVAE, where the latent space z is used as an input for the MLP.

    :param vocab_size: size of the vocabulary (default=36)
    :param latent_dim: dimensions of the latent layer (default=64)
    :param hidden_dim_vae: dimensions of the hidden layer(s) of the decoder (default=2048)
    :param kernel_size: CNN kernel size (default=8)
    :param beta: scales the KL loss (default=0.001)
    :param seq_length: length of the SMILES sequences (default=102)
    :param n_layers_mlp: number of MLP layers (including the input layer, not including the output layer, default=2)
    :param hidden_dim_mlp: hidden layer(s) dimension of the MLP (default=2048)
    :param anchored: toggles weight anchoring of the MLP (default=False)
    :param l2_lambda: L2 loss scaling for the anchored MLP loss (default=1e-4)
    :param n_ensemble: number of MLPs in the ensemble (default=10)
    :param output_dim_mlp: dimensions of the output layer of the MLP (default=2)
    :param class_scaling_factor: Scales BCE loss for the '1' class, i.e. a factor of 2 would double the respective loss
        for the '1' class (default=None)
    :param variational_scale: The scale of the Gaussian of the encoder (default=1)
    :param kwargs: just here for compatability reasons.
    """

    def __init__(self, vocab_size: int = 36, latent_dim: int = 64, hidden_dim_vae: int = 256, kernel_size: int = 8,
                 beta: float = 0.001, n_layers_mlp: int = 2, hidden_dim_mlp: int = 2048, anchored: bool = True,
                 seq_length: int = 102,  l2_lambda: float = 1e-4, n_ensemble: int = 10, output_dim_mlp: int = 2,
                 variational_scale: float = 1, device: str = None, mlp_loss_scalar: float = 1, **kwargs) -> None:
        super(LstmJVAE, self).__init__()
        self.name = 'LstmJVAE'
        self.device = device
        self.register_buffer('mlp_loss_scalar', torch.tensor(mlp_loss_scalar))

        self.vae = LstmVAE(vocab_size=vocab_size, latent_dim=latent_dim, hidden_dim=hidden_dim_vae, beta=beta,
                           kernel_size=kernel_size, variational_scale=variational_scale, seq_length=seq_length)

        self.prediction_head = Ensemble(input_dim=latent_dim, hidden_dim=hidden_dim_mlp, n_layers=n_layers_mlp,
                                        anchored=anchored, l2_lambda=l2_lambda, n_ensemble=n_ensemble,
                                        output_dim=output_dim_mlp)

    def forward(self, x: Tensor, y: Tensor = None, **kwargs) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor):

        x_hat, z, sample_likelihood, loss = self.vae(x)
        y_logits_N_K_C, loss_mlp_i, loss_mlp = self.prediction_head(z, y)

        if y is not None:
            loss = loss + self.mlp_loss_scalar * loss_mlp

        return y_logits_N_K_C, x_hat, z, sample_likelihood, loss_mlp_i, loss

    def predict(self, dataset, pretrained_vae_path: str = None, batch_size: int = 128) -> Tensor:
        return _predict_lstm_jvae(self, dataset, pretrained_vae_path=pretrained_vae_path, batch_size=batch_size)


class Decoder(nn.Module):
    """ A decoder that reconstructs a binary vector from a latent representation

    :param input_dim: dimensions of the input layer (default=2048)
    :param hidden_dim: dimensions of the hidden layer(s) (default=2048)
    :param out_dim: dimensions of the output layer (default=2)
    :param n_layers: number of layers (including the input layer, not including the output layer, default=2)
    """
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 2048, out_dim: int = 2048, n_layers: int = 1,
                 **kwargs) -> None:
        super(Decoder, self).__init__()
        self.name = 'Decoder'
        self.fc = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.fc:
            lin.reset_parameters()
        self.out.reset_parameters()

    def forward(self, z: Tensor) -> Tensor:
        for lin in self.fc:
            z = F.relu(lin(z))
        z = torch.sigmoid(self.out(z))

        return z


class VariationalEncoder(nn.Module):
    """ A simple variational encoder. Takes a batch of vectors and compresses the input space to a smaller variational
    latent.

    :param input_dim: dimensions of the input layer (default=2048)
    :param latent_dim: dimensions of the latent/output layer (default=2048)
    :param variational_scale: The scale of the Gaussian of the encoder (default=1)
    """
    def __init__(self, input_dim: int = 2048, latent_dim: int = 128, variational_scale: float = 1, **kwargs):
        super(VariationalEncoder, self).__init__()
        self.name = 'VariationalEncoder'

        self.lin0_x = nn.Linear(input_dim, latent_dim)
        self.lin0_mu = nn.Linear(latent_dim, latent_dim)
        self.lin0_sigma = nn.Linear(latent_dim, latent_dim)

        self.N = torch.distributions.Normal(0, variational_scale)
        self.N.loc = self.N.loc
        self.N.scale = self.N.scale
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.lin0_x(x))
        mu = self.lin0_mu(x)
        sigma = torch.exp(self.lin0_sigma(x))

        # reparameterization trick
        z = mu + sigma * self.N.sample(mu.shape).to('cuda' if torch.cuda.is_available() else 'cpu')  # TODO fix this
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()

        return z


@torch.no_grad()
def _predict_vae(model, dataset, batch_size: int = 128) -> (Tensor, Tensor, Tensor):
    """ Get predictions from a dataloader

    :param model: torch module (e.g. MLP or Ensemble)
    :param dataset: dataset of the data to predict; jcm.datasets.MoleculeDataset
    :param batch_size: prediction batch size (default=128)

    :return: logits_N_K_C, vae latents, vae likelihoods
    """
    val_loader = DataLoader(dataset, sampler=None, shuffle=False, pin_memory=True, batch_size=batch_size,
                            collate_fn=single_batchitem_fix)
    y_hats = []
    zs = []
    sample_likelihoods = []

    model.eval()
    for x, y in val_loader:
        # move to device
        x.to(model.device)

        # predict
        y_hat, z, sample_likelihood, loss = model(x)

        y_hats.append(y_hat)
        zs.append(z)
        sample_likelihoods.append(sample_likelihood)

    model.train()

    y_hats = torch.cat(y_hats, 0)
    zs = torch.cat(zs)
    sample_likelihoods = torch.cat(sample_likelihoods, 0)

    return y_hats, zs, sample_likelihoods


@torch.no_grad()
def _predict_jvae(model, dataset, pretrained_vae_path: str = None, batch_size: int = 128) -> (Tensor, Tensor, Tensor):
    """ Get predictions from a dataloader

    :param model: torch module (e.g. MLP or Ensemble)
    :param dataset: dataset of the data to predict; jcm.datasets.MoleculeDataset
    :param pretrained_vae_path: path of the pretrained EcfpVAE, used to normalize likelihoods if supplied (default=None)
    :param batch_size: prediction batch size (default=128)

    :return: logits_N_K_C, x_reconstructions, vae latents, vae likelihoods
    """

    if pretrained_vae_path is not None:
        config = Config()
        # config.set_hyperparameters(**VAE_PRETRAIN_HYPERPARAMETERS)    TODO FIX THIS

        pre_trained_vae = LstmVAE(**config.hyperparameters)
        pre_trained_vae.load_state_dict(torch.load(pretrained_vae_path))
        pre_trained_vae.eval()

    val_loader = DataLoader(dataset, sampler=None, shuffle=False, pin_memory=True, batch_size=batch_size,
                            collate_fn=single_batchitem_fix)

    y_hats = []
    x_hats = []
    zs = []
    sample_likelihoods = []
    pt_sample_likelihoods = []

    model.eval()
    for x, y in val_loader:
        # move to device
        x.to(model.device)

        # predict
        y_hat, x_hat, z, sample_likelihood, loss = model(x)

        if pretrained_vae_path is not None:
            pt_x_hat, pt_z, pt_sample_likelihood, pt_loss = pre_trained_vae(x)

        y_hats.append(y_hat)
        zs.append(z)
        x_hats.append(x_hat)
        sample_likelihoods.append(sample_likelihood)
        pt_sample_likelihoods.append(pt_sample_likelihood)

    model.train()

    y_hats = torch.cat(y_hats, 0)
    x_hats = torch.cat(x_hats)
    zs = torch.cat(zs)
    sample_likelihoods = torch.cat(sample_likelihoods, 0)

    # normalize X likelihoods using the pretrained likelihoods
    if pretrained_vae_path is not None:
        pt_sample_likelihoods = torch.cat(pt_sample_likelihoods, 0)
        sample_likelihoods = sample_likelihoods - pt_sample_likelihoods

    return y_hats, x_hats, zs, sample_likelihoods


@torch.no_grad()
def _predict_lstm_vae(model, dataset, batch_size: int = 128) -> (Tensor, Tensor, Tensor):
    """ Get predictions from a dataloader

    :param model: torch module (e.g. MLP or Ensemble)
    :param dataset: dataset of the data to predict; jcm.datasets.MoleculeDataset
    :param batch_size: prediction batch size (default=128)

    :return: logits_N_K_C, vae latents, vae likelihoods
    """
    val_loader = DataLoader(dataset, sampler=None, shuffle=False, pin_memory=True, batch_size=batch_size,
                            collate_fn=single_batchitem_fix)
    x_hats = []
    zs = []
    sample_likelihoods = []

    model.eval()
    for x in val_loader:
        # move to device
        x = x.to(model.device)

        # predict
        x_hat, z, sample_likelihood, loss = model(x)

        x_hats.append(x_hat.cpu())
        zs.append(z.cpu())
        sample_likelihoods.append(sample_likelihood.cpu())

    model.train()

    y_hats = torch.cat(x_hats, 0)
    zs = torch.cat(zs)
    sample_likelihoods = torch.cat(sample_likelihoods, 0)

    return y_hats, zs, sample_likelihoods


@torch.no_grad()
def _predict_lstm_jvae(model, dataset, pretrained_vae_path: str = None, batch_size: int = 128) -> \
        (Tensor, Tensor, Tensor):
    """ Get predictions from a dataloader

    :param model: torch module (e.g. MLP or Ensemble)
    :param dataset: dataset of the data to predict; jcm.datasets.MoleculeDataset
    :param pretrained_vae_path: path of the pretrained EcfpVAE, used to normalize likelihoods if supplied (default=None)
    :param batch_size: prediction batch size (default=128)

    :return: logits_N_K_C, x_reconstructions, vae latents, vae likelihoods
    """

    if pretrained_vae_path is not None:
        config = Config()
        # config.set_hyperparameters(**VAE_PRETRAIN_HYPERPARAMETERS)  TODO FIX THIS

        pre_trained_vae = LstmVAE(**config.hyperparameters)
        pre_trained_vae.load_state_dict(torch.load(pretrained_vae_path))
        pre_trained_vae.eval()

    val_loader = DataLoader(dataset, sampler=None, shuffle=False, pin_memory=True, batch_size=batch_size,
                            collate_fn=single_batchitem_fix)

    y_hats = []
    x_hats = []
    zs = []
    sample_likelihoods = []
    pt_sample_likelihoods = []

    model.eval()
    for x, y in val_loader:
        # move to device
        x.to(model.device)

        # predict
        y_hat, x_hat, z, sample_likelihood, loss = model(x.long())

        if pretrained_vae_path is not None:
            pt_x_hat, pt_z, pt_sample_likelihood, pt_loss = pre_trained_vae(x.long())

        y_hats.append(y_hat)
        zs.append(z)
        x_hats.append(x_hat)
        sample_likelihoods.append(sample_likelihood)
        pt_sample_likelihoods.append(pt_sample_likelihood)

    model.train()

    y_hats = torch.cat(y_hats, 0)
    x_hats = torch.cat(x_hats)
    zs = torch.cat(zs)
    sample_likelihoods = torch.cat(sample_likelihoods, 0)

    # normalize X likelihoods using the pretrained likelihoods
    if pretrained_vae_path is not None:
        pt_sample_likelihoods = torch.cat(pt_sample_likelihoods, 0)
        sample_likelihoods = sample_likelihoods - pt_sample_likelihoods

    return y_hats, x_hats, zs, sample_likelihoods


def token_loss(logits: Tensor, target: Tensor, ignore_index: int = -1, weight: Tensor = None):
    """

    :param logits: model output in the shape of (batch_size, sequence_length, vocab)
    :param target: target token indices in the shape of (batch_size, sequence_length)
    :param ignore_index: token to ignore
    :param weight: tensor with shape C that weights all tokens (default = None, 1 = standard weight)
    :return:
    """
    loss_func = nn.NLLLoss(reduction='none', ignore_index=ignore_index, weight=weight)

    x_hat = F.log_softmax(logits, dim=-1)

    loss = loss_func(x_hat.transpose(2, 1), target[:, 1:])

    # Considering that the padding token is the last token in the vocab, we can use argmax to find the first
    # occurence of this highest number.
    length_of_smiles = torch.argmax(target, 1) + 1
    # devide the summed loss per token by the sequence length
    sample_loss = torch.sum(loss, 1) / length_of_smiles

    return sample_loss, torch.mean(sample_loss)


    # Considering that the padding token is the last token in the vocab, we can use argmax to find the first
    # occurence of this highest number.
    length_of_smiles = torch.argmax(target, 1) + 1
    # devide the summed loss per token by the sequence length
    sample_loss = torch.sum(loss, 1) / length_of_smiles

    return sample_loss, torch.mean(sample_loss)
