
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data.dataloader import DataLoader
from dataprep.descriptors import one_hot_encode, encoding_to_smiles, mols_to_maccs, mols_to_ecfp
from jcm.utils import BCE_per_sample, single_batchitem_fix, calc_l_out
from jcm.config import Config
from constants import VAE_PRETRAIN_HYPERPARAMETERS, VOCAB
from rdkit import Chem


class CnnEncoder(nn.Module):
    """ Encode a one-hot encoded SMILES string with a CNN. Uses Max Pooling and flattens conv layer at the end

    :param channels: vocab size (default=40)
    :param seq_length: sequence length of SMILES strings (default=32)
    :param out_hidden: dimension of the CNN token embedding size (default=256)
    :param kernel_size: CNN kernel_size (default=8)
    :param stride: stride (default=1)
    """

    def __init__(self, channels: int = 40, seq_length: int = 32, out_hidden: int = 256, kernel_size: int = 8,
                 stride: int = 1, **kwargs):
        super().__init__()

        self.cnn0 = nn.Conv1d(channels, 64, kernel_size=kernel_size, stride=stride)
        self.cnn1 = nn.Conv1d(64, 128, kernel_size=kernel_size, stride=stride)
        self.cnn2 = nn.Conv1d(128, 256, kernel_size=kernel_size, stride=stride)
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

        self.l_out = calc_l_out(seq_length, self.cnn0, self.pool, self.cnn1, self.pool, self.cnn2, self.pool)
        self.out_dim = int(out_hidden * self.l_out)

    def forward(self, x):
        x = F.relu(self.cnn0(x))
        x = self.pool(x)
        x = F.relu(self.cnn1(x))
        x = self.pool(x)
        x = F.relu(self.cnn2(x))
        x = self.pool(x)

        # flatten
        x = x.view(x.size(0), -1)

        return x


class LSTMDecoder(nn.Module):
    """ An autoregressive GRU that decodes a batch of latent vectors (batch_size, embedding_size) into
    a SMILES strings (batch_size, vocab_size, sequence_length)

    :param hidden_size: size of the hidden layers in the LSTM (default=64)
    :param vocabulary_size: number of tokens in the vocab (default=40)
    :param sequence_length: length of the SMILES strings (default=32)
    :param device: device (can be 'cpu' or 'cuda')
    """

    def __init__(self, hidden_size: int, vocabulary_size: int, sequence_length: int, device: str, **kwargs):
        super(LSTMDecoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length

        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, z, sequence_length: int = None) -> Tensor:

        batch_size = z.size(0)
        sequence_length = self.sequence_length if sequence_length is None else sequence_length

        # initiate the hidden state (the latent embedding) and the cell state (fresh)
        hidden_state = torch.zeros(batch_size, self.hidden_size).to(self.device)
        cell_state = torch.zeros(batch_size, self.hidden_size).to(self.device)

        # autoregress
        output = []
        for i in range(sequence_length):
            hidden_state, cell_state = self.lstm(z, (hidden_state, cell_state))
            output.append(hidden_state)

        x = torch.stack(output, 1)
        x = F.relu(self.fc(x))

        return x


class GruDecoder(nn.Module):
    """ An autoregressive GRU that decodes a batch of latent vectors (batch_size, embedding_size) into
    a SMILES strings (batch_size, vocab_size, sequence_length)

    :param hidden_size: size of the hidden layers in the LSTM (default=64)
    :param vocabulary_size: number of tokens in the vocab (default=40)
    :param sequence_length: length of the SMILES strings (default=32)
    :param device: device (can be 'cpu' or 'cuda')
    """

    def __init__(self, hidden_size: int, vocabulary_size: int, sequence_length: int, device: str, **kwargs):
        super(GruDecoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length

        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, z, sequence_length: int = None) -> Tensor:

        batch_size = z.size(0)
        sequence_length = self.sequence_length if sequence_length is None else sequence_length

        # initiate the hidden state (the latent embedding) and the cell state (fresh)
        hidden_state = torch.zeros(batch_size, self.hidden_size).to(self.device)

        # autoregress
        output = []
        for i in range(sequence_length):
            hidden_state = self.gru(z, hidden_state)
            output.append(hidden_state)

        x = torch.stack(output, 1)
        x = F.relu(self.fc(x))

        return x


class LstmECFP(nn.Module):
    def __init__(self, hidden_size: int, vocabulary_size: int = 40, sequence_length: int = 32, bitsize: int = 512, device: str = 'cpu',
                 **kwargs):
        super(LstmECFP, self).__init__()
        self.device = device
        self.bitsize = bitsize
        self.lstm = LSTMDecoder(hidden_size, vocabulary_size, sequence_length, device)
        self.lin = nn.Linear(bitsize, hidden_size)

    def forward(self, x, y=None):

        mols = [Chem.MolFromSmiles(encoding_to_smiles(smi.tolist())) for smi in x]
        # z = torch.tensor(mols_to_maccs(mols, to_array=True)).float()
        z = torch.tensor(mols_to_ecfp(mols, nbits=self.bitsize, to_array=True)).float()
        z_ = F.relu(self.lin(z))

        x_hat = self.lstm(z_)
        # x = one_hot_encode(x.long()) #.transpose(2, 1)

        # compute losses
        sample_likelihood, loss = token_loss(x_hat, x.long())

        return x_hat, z, sample_likelihood, loss


class LstmVAE(nn.Module):
    """ A  LstmVAE, where the latent space z is used as an input for the MLP.

    :param vocab_size: size of the vocabulary (default=40)
    :param latent_dim: dimensions of the latent layer (default=64)
    :param hidden_dim: dimensions of the hidden layer(s) of the CNN encoder (default=256)
    :param kernel_size: CNN kernel size (default=8)
    :param beta: scales the KL loss (default=0.001)
    :param seq_length: length of the SMILES sequences (default=32)
    :param variational_scale: The scale of the Gaussian of the encoder (default=1)
    :param device: device (default=None, can be 'cuda' or 'cpu')
    :param kwargs: Just here for compatability
    """

    def __init__(self, vocab_size: int = 40, latent_dim: int = 128, hidden_dim: int = 256, kernel_size: int = 8,
                 beta: float = 0.001, seq_length: int = 32, variational_scale: float = 1, device: str = None, **kwargs):
        super(LstmVAE, self).__init__()
        self.name = 'LstmVAE'
        self.device = device
        self.register_buffer('beta', torch.tensor(beta))

        self.cnn = CnnEncoder(channels=vocab_size, seq_length=seq_length, out_hidden=hidden_dim,
                              kernel_size=kernel_size)
        self.variational_layer = VariationalEncoder(input_dim=self.cnn.out_dim, latent_dim=latent_dim,
                                                    variational_scale=variational_scale)
        self.decoder = LSTMDecoder(latent_dim, vocab_size, seq_length, device=self.device)

    def forward(self, x: Tensor, y: Tensor = None) -> (Tensor, Tensor, Tensor, Tensor):

        # turn indexed encoding into one-hots w. shape N, C, L
        x_ = one_hot_encode(x.long()).transpose(1, 2).float()   # TODO fix shapes?

        # Get latent vectors and decode them back into a molecule
        z = self.variational_layer(self.cnn(x_))
        x_hat = self.decoder(z)

        # compute losses
        sample_likelihood, loss_reconstruction = token_loss(x_hat, x.long())
        loss_kl = self.variational_layer.kl / x.shape[0]  # divide by batch size
        loss = loss_reconstruction + self.beta * loss_kl  # add the reconstruction loss and the scaled KL loss

        return x_hat, z, sample_likelihood, loss

    def predict(self, dataset, batch_size: int = 128) -> Tensor:
        return _predict_lstm_vae(self, dataset, batch_size)


class LstmJVAE(nn.Module):
    """ A joint LstmVAE, where the latent space z is used as an input for the MLP.

    :param vocab_size: size of the vocabulary (default=39)
    :param latent_dim: dimensions of the latent layer (default=64)
    :param hidden_dim_vae: dimensions of the hidden layer(s) of the decoder (default=2048)
    :param kernel_size: CNN kernel size (default=8)
    :param beta: scales the KL loss (default=0.001)
    :param seq_length: length of the SMILES sequences (default=100)
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

    def __init__(self, vocab_size: int = 40, latent_dim: int = 64, hidden_dim_vae: int = 256, kernel_size: int = 8,
                 beta: float = 0.001, n_layers_mlp: int = 2, hidden_dim_mlp: int = 2048, anchored: bool = True,
                 seq_length: int = 32,  l2_lambda: float = 1e-4, n_ensemble: int = 10, output_dim_mlp: int = 2,
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

    def forward(self, x: Tensor, y: Tensor = None, **kwargs):

        x_hat, z, sample_likelihood, loss = self.vae(x)
        y_logits_N_K_C, loss_mlp_i, loss_mlp = self.prediction_head(z, y)

        if y is not None:
            loss = loss + self.mlp_loss_scalar * loss_mlp

        return y_logits_N_K_C, x_hat, z, sample_likelihood, loss_mlp_i, loss

    def predict(self, dataset, pretrained_vae_path: str = None, batch_size: int = 128) -> Tensor:
        return _predict_lstm_jvae(self, dataset, pretrained_vae_path=pretrained_vae_path, batch_size=batch_size)


class MLP(nn.Module):
    """ Multi-Layer Perceptron with weight anchoring according to Pearce et al. (2018)

    :param input_dim: input layer dimension (default=2048)
    :param hidden_dim: hidden layer(s) dimension (default=2048)
    :param output_dim: output layer dimension (default=2)
    :param n_layers: number of layers (including the input layer, not including the output layer, default=2)
    :param seed: random seed (default=42)
    :param anchored: toggles weight anchoring (default=False)
    :param l2_lambda: L2 loss scaling for the anchored loss (default=1e-4)
    :param device: 'cpu' or 'cuda' (default=None)
    """
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 2, n_layers: int = 2,
                 seed: int = 42, anchored: bool = False, l2_lambda: float = 1e-4, device: str = None, **kwargs) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.l2_lambda = l2_lambda
        self.anchored = anchored
        self.name = 'MLP'
        self.device = device

        self.fc = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(AnchoredLinear(input_dim if i == 0 else hidden_dim, hidden_dim, device=self.device))
        self.out = AnchoredLinear(hidden_dim, output_dim, device=self.device)

    def reset_parameters(self):
        for lin in self.fc:
            lin.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor, y: Tensor = None) -> (Tensor, Tensor):
        for lin in self.fc:
            x = F.relu(lin(x))
        x = self.out(x)
        x = F.log_softmax(x, 1)

        loss_i, loss = anchored_loss(self, x, y)

        return x, loss_i, loss

    def predict(self, dataset, batch_size: int = 128) -> Tensor:
        return _predict_mlp(self, dataset, batch_size)


class AnchoredLinear(nn.Module):
    """ Applies a linear transformation to the incoming data: :math:`y = xA^T + b` and stores original init weights as
    a buffer for regularization later on.

    :param in_features: size of each input sample
    :param out_features: size of each output sample
    :param bias: If set to False, the layer will not learn an additive bias. (default=True)
    :param device: 'cpu' or 'cuda' (default=None)
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        self.weight = Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # store the init weight/bias as a buffer
        self.register_buffer('anchor_weight', self.weight.clone().detach())
        self.register_buffer('anchor_bias', self.bias.clone().detach())

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)


class EnsembleFrame(nn.Module):
    """ An ensemble of (anchored) MLPs, used for uncertainty estimation. Outputs logits_N_K_C (n_ensemble, batch_size,
    classes) and a (regularized) NLL loss

    :param input_dim: dimensions of the input layer (default=2048)
    :param hidden_dim: dimensions of the hidden layer(s) (default=2048)
    :param output_dim: dimensions of the output layer (default=2)
    :param n_layers: number of layers (including the input layer, not including the output layer, default=2)
    :param anchored: toggles the use of anchored loss regularization, Pearce et al. (2018) (default=True)
    :param l2_lambda: L2 loss scaling for the anchored loss (default=1e-4)
    :param n_ensemble: number of models in the ensemble (default=10)
    :param device: 'cpu' or 'cuda' (default=None)
    """
    def __init__(self, input_dim: int = 2048, latent_dim: int = 64, hidden_dim: int = 2048, output_dim: int = 2,
                 n_layers: int = 2, anchored: bool = True, l2_lambda: float = 1e-4, n_ensemble: int = 10,
                 device: str = None, **kwargs) -> None:
        super().__init__()
        self.name = 'EnsembleFrame'
        self.device = device

        # latent_dim
        self.compress = nn.Linear(input_dim, latent_dim)
        self.ensemble = Ensemble(input_dim=latent_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers,
                                 anchored=anchored, l2_lambda=l2_lambda, n_ensemble=n_ensemble, device=device)

    def forward(self, x: Tensor, y: Tensor = None):

        x = F.relu(self.compress(x))
        logits_N_K_C, loss = self.ensemble(x, y)

        return logits_N_K_C, loss

    def predict(self, dataset, batch_size: int = 128) -> Tensor:
        return _predict_mlp(self, dataset, batch_size)


class Ensemble(nn.Module):
    """ An ensemble of (anchored) MLPs, used for uncertainty estimation. Outputs logits_N_K_C (n_ensemble, batch_size,
    classes) and a (regularized) NLL loss

    :param input_dim: dimensions of the input layer (default=2048)
    :param hidden_dim: dimensions of the hidden layer(s) (default=2048)
    :param output_dim: dimensions of the output layer (default=2)
    :param n_layers: number of layers (including the input layer, not including the output layer, default=2)
    :param anchored: toggles the use of anchored loss regularization, Pearce et al. (2018) (default=True)
    :param l2_lambda: L2 loss scaling for the anchored loss (default=1e-4)
    :param n_ensemble: number of models in the ensemble (default=10)
    :param device: 'cpu' or 'cuda' (default=None)
    """
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 2, n_layers: int = 2,
                 anchored: bool = True, l2_lambda: float = 1e-4, n_ensemble: int = 10, device: str = None,
                 **kwargs) -> None:
        super().__init__()
        self.name = 'Ensemble'
        self.device = device

        self.mlps = nn.ModuleList()
        for i in range(n_ensemble):
            self.mlps.append(MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers,
                                 seed=i, anchored=anchored, l2_lambda=l2_lambda, device=device))

    def forward(self, x: Tensor, y: Tensor = None):

        loss = Tensor([0])
        y_hats = []
        loss_items = []
        for mlp_i in self.mlps:
            y_hat_i, loss_item_i, loss_i = mlp_i(x, y)
            y_hats.append(y_hat_i)
            loss_items.append(loss_item_i)
            if loss_i is not None:
                loss += loss_i

        # Compute the mean losses over the ensemble. Both the total loss and the item-wise loss
        loss = None if y is None else loss/len(self.mlps)
        loss_items = None if y is None else torch.mean(torch.stack(loss_items), 0)
        logits_N_K_C = torch.stack(y_hats).permute(1, 0, 2)

        return logits_N_K_C, loss_items, loss

    def predict(self, dataset, batch_size: int = 128) -> Tensor:
        return _predict_mlp(self, dataset, batch_size)


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


class EcfpVAE(nn.Module):
    """ a Variational Autoencoder, returns: (logits_N_K_C, vae_latents, vae_likelihoods, loss)

    :param input_dim: dimensions of the input layer (default=2048)
    :param latent_dim: dimensions of the latentlayer (default=128)
    :param hidden_dim: dimensions of the hidden layer(s) of the decoder (default=2048)
    :param out_dim: dimensions of the output layer (default=2048)
    :param beta: scales the KL loss (default=0.001)
    :param class_scaling_factor: Scales BCE loss for the '1' class, i.e. a factor of 2 would double the respective loss
        for the '1' class (default=None)
    :param variational_scale: The scale of the Gaussian of the encoder (default=1)
    :param kwargs: just here for compatability reasons.
    """
    def __init__(self, input_dim: int = 2048, latent_dim: int = 128, hidden_dim: int = 2048, out_dim: int = 2048,
                 beta: float = 0.001, class_scaling_factor: float = 1, variational_scale: float = 1, device: str = None,
                 **kwargs):
        super(EcfpVAE, self).__init__()
        self.name = 'EcfpVAE'
        self.device = device

        self.register_buffer('beta', torch.tensor(beta))
        self.register_buffer('class_scaling_factor', torch.tensor(class_scaling_factor))
        self.encoder = VariationalEncoder(input_dim=input_dim, latent_dim=latent_dim,
                                          variational_scale=variational_scale)
        self.decoder = Decoder(input_dim=latent_dim, hidden_dim=hidden_dim, out_dim=out_dim)

    def forward(self, x: Tensor, y: Tensor = None):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        # compute losses
        loss_reconstruction, sample_likelihood = BCE_per_sample(x_hat, x, self.class_scaling_factor)
        loss_kl = self.encoder.kl / x.shape[0]
        loss = loss_reconstruction + self.beta * loss_kl

        return x_hat, z, sample_likelihood, loss

    def predict(self, dataset, batch_size: int = 128) -> Tensor:
        return _predict_vae(self, dataset, batch_size)


class EcfpJVAE(nn.Module):
    """ A joint EcfpVAE, where the latent space z is used as an input for the MLP.

    :param input_dim: dimensions of the input layer (default=2048)
    :param latent_dim: dimensions of the latent layer (default=128)
    :param hidden_dim_vae: dimensions of the hidden layer(s) of the decoder (default=2048)
    :param out_dim_vae: dimensions of the output layer (default=2048)
    :param beta: scales the KL loss (default=0.001)
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
    def __init__(self, input_dim: int = 2048, latent_dim: int = 32, hidden_dim_vae: int = 2048, out_dim_vae: int = 2048,
                 beta: float = 0.001, n_layers_mlp: int = 2, hidden_dim_mlp: int = 2048, anchored: bool = True,
                 l2_lambda: float = 1e-4, n_ensemble: int = 10, output_dim_mlp: int = 2, class_scaling_factor: float = 1,
                 variational_scale: float = 1, device: str = None, mlp_loss_scalar: float = 1, **kwargs) -> None:
        super(EcfpJVAE, self).__init__()
        self.name = 'EcfpJVAE'
        self.device = device
        self.register_buffer('mlp_loss_scalar', torch.tensor(mlp_loss_scalar))

        self.vae = EcfpVAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim_vae, out_dim=out_dim_vae,
                           beta=beta, class_scaling_factor=class_scaling_factor, variational_scale=variational_scale)
        self.prediction_head = Ensemble(input_dim=latent_dim, hidden_dim=hidden_dim_mlp, n_layers=n_layers_mlp,
                                        anchored=anchored, l2_lambda=l2_lambda, n_ensemble=n_ensemble,
                                        output_dim=output_dim_mlp)

    def forward(self, x: Tensor, y: Tensor = None, **kwargs):
        x_hat, z, sample_likelihood, loss = self.vae(x)
        y_logits_N_K_C, loss_mlp_i, loss_mlp = self.prediction_head(z, y)

        if y is not None:
            loss = loss + self.mlp_loss_scalar * loss_mlp

        return y_logits_N_K_C, x_hat, z, sample_likelihood, loss_mlp_i, loss

    def predict(self, dataset, pretrained_vae_path: str = None, batch_size: int = 128) -> Tensor:
        return _predict_jvae(self, dataset, pretrained_vae_path=pretrained_vae_path, batch_size=batch_size)


def anchored_loss(model: MLP, x: Tensor, y: Tensor = None) -> Tensor:
    """ Compute anchored loss according to Pearce et al. (2018)

    :param model: MLP torch module
    :param x: model predictions
    :param y: target tensor (default = None)
    :return: (loss_i, loss) or (None, None) (if y is None)
    """
    if y is None:
        return None, None

    loss_func = torch.nn.NLLLoss(reduction='none')
    loss_i = loss_func(x, y)

    if model.anchored:
        l2_loss = 0
        for p, p_a in zip(model.named_parameters(), model.named_buffers()):
            assert p_a[1].shape == p[1].shape
            l2_loss += model.l2_lambda * torch.mul(p[1] - p_a[1], p[1] - p_a[1]).sum()

        loss_i = loss_i + l2_loss/len(y)

    loss = torch.mean(loss_i)

    return loss_i, loss


@torch.no_grad()
def _predict_mlp(model, dataset, batch_size: int = 128) -> Tensor:
    """ Get predictions from a dataloader

    :param model: torch module (e.g. MLP or Ensemble)
    :param dataset: dataset of the data to predict; jcm.datasets.MoleculeDataset
    :param batch_size: prediction batch size (default=128)

    :return: logits_N_K_C, vae latents, vae likelihoods
    """
    val_loader = DataLoader(dataset, sampler=None, shuffle=False, pin_memory=True, batch_size=batch_size,
                            collate_fn=single_batchitem_fix)
    y_hats = []

    model.eval()
    for x, y in val_loader:
        # move to device
        x.to(model.device)

        # predict
        y_hat, loss = model(x)      # x_hat, z, sample_likelihood, loss
        y_hats.append(y_hat)

    model.train()

    return torch.cat(y_hats, 0)


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
        config.set_hyperparameters(**VAE_PRETRAIN_HYPERPARAMETERS)

        pre_trained_vae = EcfpVAE(**config.hyperparameters)
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
    for x, y in val_loader:
        # move to device
        x.to(model.device)

        # predict
        x_hat, z, sample_likelihood, loss = model(x.long())

        x_hats.append(x_hat)
        zs.append(z)
        sample_likelihoods.append(sample_likelihood)

    model.train()

    y_hats = torch.cat(x_hats, 0)
    zs = torch.cat(zs)
    sample_likelihoods = torch.cat(sample_likelihoods, 0)

    return y_hats, zs, sample_likelihoods


@torch.no_grad()
def _predict_lstm_jvae(model, dataset, pretrained_vae_path: str = None, batch_size: int = 128) -> (Tensor, Tensor, Tensor):
    """ Get predictions from a dataloader

    :param model: torch module (e.g. MLP or Ensemble)
    :param dataset: dataset of the data to predict; jcm.datasets.MoleculeDataset
    :param pretrained_vae_path: path of the pretrained EcfpVAE, used to normalize likelihoods if supplied (default=None)
    :param batch_size: prediction batch size (default=128)

    :return: logits_N_K_C, x_reconstructions, vae latents, vae likelihoods
    """

    if pretrained_vae_path is not None:
        config = Config()
        config.set_hyperparameters(**VAE_PRETRAIN_HYPERPARAMETERS)

        pre_trained_vae = EcfpVAE(**config.hyperparameters)
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


def lstm_loss(x_hat: Tensor, x: Tensor, padding_idx: int = -1) -> (Tensor, Tensor):

    # remove first token and convert (batch_size, vocab, seq_length) to (batch_size, seq_length) with argmax indices
    x_no_start_token = x[:, :, 1:].argmax(1)

    # if padding idx is -1, take the tast token from the target sequence.
    if padding_idx == -1:
        padding_idx = x_no_start_token[0][-1].item()

    # calculate the loss for every token -> (batch_size, seq_length), the padding tokens will have a 0 loss
    loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=padding_idx)
    sample_loss = loss_fn(x_hat, x_no_start_token)

    # Considering that the padding token is the last token in the vocab, we can use argmax to find the first
    # occurence of this highest number.
    length_of_smiles = torch.argmax(x_no_start_token, 1)
    # devide the summed loss per token by the sequence length
    sample_loss = torch.sum(sample_loss, 1) / length_of_smiles

    return torch.mean(sample_loss), sample_loss  # taking the mean of the sample loss equals using mean reduction



# #####
#
#
# x = torch.rand((128, 39, 100))  # N, C, L
#
# cnn = CnnEncoder()
# var = VariationalEncoder(cnn.out_dim, 64)
#
# z = var(cnn(x))
# z.shape
#
# z      # N, L, H
#
# lstm = nn.LSTMCell(39, 64)
# fc = nn.Linear(64, 39)
#
# hidden_state = torch.zeros(128, 64)
# cell_state = torch.zeros(128, 64)
#
# current_token = torch.rand((128, 39))
#
# outputs
# hidden_state, cell_state = lstm(current_token, (hidden_state, cell_state))
# out = fc(hidden_state)
#
# current_token = out.argmax(dim=1)
# current_token = output.argmax(dim=1)
#
# out.shape
#
# out.shape
#
# class AutoregressiveLSTM(nn.Module):
#     def __init__(self, hidden_size, vocabulary_size, sequence_length):
#         super(AutoregressiveLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.vocabulary_size = vocabulary_size
#         self.sequence_length = sequence_length
#
#         self.lstm = nn.LSTMCell(vocabulary_size, hidden_size)
#         self.fc = nn.Linear(hidden_size, vocabulary_size)
#
#     def forward(self, encoder_output, start_token, steps=10):
#         batch_size = encoder_output.size(0)
#         current_token = start_token
#         hidden_state = torch.zeros(batch_size, self.hidden_size).to(encoder_output.device)
#         cell_state = torch.zeros(batch_size, self.hidden_size).to(encoder_output.device)
#         outputs = []
#
#         for _ in range(steps):
#             hidden_state, cell_state = self.lstm(current_token, (hidden_state, cell_state))
#             output = self.fc(hidden_state)
#             outputs.append(output.unsqueeze(1))
#             current_token = output.argmax(dim=1)  # Use argmax as next input
#         outputs = torch.cat(outputs, dim=1)
#         return outputs
#
#
#
# lstm = nn.LSTM(input_size=64, hidden_size=256, num_layers=1, batch_first=True)
#
# # Initialize LSTM hidden state and cell state
# h0 = torch.zeros(128, 1, 256)  # N L H
# c0 = torch.zeros(128, 39, 256)
#
# x_hat, _ = lstm(x, (h0, c0))
#
# x_hat.shape
#
# transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, batch_first=True)
#
# src = torch.rand((32, 1, 512))
# tgt = torch.rand((32, 39, 512))
# out = transformer_model(src, tgt)
# out.shape
