
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data.dataloader import DataLoader
from jcm.utils import BCE_per_sample
from jcm.config import Config
from constants import VAE_PRETRAIN_HYPERPARAMETERS


class MLP(nn.Module):
    """ Multi-Layer Perceptron with weight anchoring according to Pearce et al. (2018)

    :param input_dim: input layer dimension (default=1024)
    :param hidden_dim: hidden layer(s) dimension (default=1024)
    :param output_dim: output layer dimension (default=2)
    :param n_layers: number of layers (including the input layer, not including the output layer, default=2)
    :param seed: random seed (default=42)
    :param anchored: toggles weight anchoring (default=False)
    :param l2_lambda: L2 loss scaling for the anchored loss (default=1e-4)
    :param device: 'cpu' or 'cuda' (default=None)
    """
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 1024, output_dim: int = 2, n_layers: int = 2,
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

        loss = anchored_loss(self, x, y)

        return x, loss

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


class Ensemble(nn.Module):
    """ An ensemble of (anchored) MLPs, used for uncertainty estimation. Outputs logits_N_K_C (n_ensemble, batch_size,
    classes) and a (regularized) NLL loss

    :param input_dim: dimensions of the input layer (default=1024)
    :param hidden_dim: dimensions of the hidden layer(s) (default=1024)
    :param output_dim: dimensions of the output layer (default=2)
    :param n_layers: number of layers (including the input layer, not including the output layer, default=2)
    :param anchored: toggles the use of anchored loss regularization, Pearce et al. (2018) (default=True)
    :param l2_lambda: L2 loss scaling for the anchored loss (default=1e-4)
    :param n_ensemble: number of models in the ensemble (default=10)
    :param device: 'cpu' or 'cuda' (default=None)
    """
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 1024, output_dim: int = 2, n_layers: int = 2,
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
        for mlp_i in self.mlps:
            y_hat_i, loss_i = mlp_i(x, y)
            y_hats.append(y_hat_i)
            if loss_i is not None:
                loss += loss_i

        loss = None if y is None else loss/len(self.mlps)
        logits_N_K_C = torch.stack(y_hats).permute(1, 0, 2)

        return logits_N_K_C, loss

    def predict(self, dataset, batch_size: int = 128) -> Tensor:
        return _predict_mlp(self, dataset, batch_size)


class Decoder(nn.Module):
    """ A decoder that reconstructs a binary vector from a latent representation

    :param input_dim: dimensions of the input layer (default=1024)
    :param hidden_dim: dimensions of the hidden layer(s) (default=1024)
    :param out_dim: dimensions of the output layer (default=2)
    :param n_layers: number of layers (including the input layer, not including the output layer, default=2)
    """
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 1024, out_dim: int = 1024, n_layers: int = 1,
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

    :param input_dim: dimensions of the input layer (default=1024)
    :param latent_dim: dimensions of the latent/output layer (default=1024)
    :param variational_scale: The scale of the Gaussian of the encoder (default=1)
    """
    def __init__(self, input_dim: int = 1024, latent_dim: int = 128, variational_scale: float = 1, **kwargs):
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


class VAE(nn.Module):
    """ a Variational Autoencoder, returns: (logits_N_K_C, vae_latents, vae_likelihoods, loss)

    :param input_dim: dimensions of the input layer (default=1024)
    :param latent_dim: dimensions of the latentlayer (default=128)
    :param hidden_dim: dimensions of the hidden layer(s) of the decoder (default=1024)
    :param out_dim: dimensions of the output layer (default=1024)
    :param beta: scales the KL loss (default=0.001)
    :param class_scaling_factor: Scales BCE loss for the '1' class, i.e. a factor of 2 would double the respective loss
        for the '1' class (default=None)
    :param variational_scale: The scale of the Gaussian of the encoder (default=1)
    :param kwargs: just here for compatability reasons.
    """
    def __init__(self, input_dim: int = 1024, latent_dim: int = 128, hidden_dim: int = 1024, out_dim: int = 1024,
                 beta: float = 0.001, class_scaling_factor: float = 1, variational_scale: float = 1, device: str = None,
                 **kwargs):
        super(VAE, self).__init__()
        self.name = 'VAE'
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


class JVAE(nn.Module):
    """ A joint VAE, where the latent space z is used as an input for the MLP.

    :param input_dim: dimensions of the input layer (default=1024)
    :param latent_dim: dimensions of the latent layer (default=128)
    :param hidden_dim_vae: dimensions of the hidden layer(s) of the decoder (default=1024)
    :param out_dim_vae: dimensions of the output layer (default=1024)
    :param beta: scales the KL loss (default=0.001)
    :param n_layers_mlp: number of MLP layers (including the input layer, not including the output layer, default=2)
    :param hidden_dim_mlp: hidden layer(s) dimension of the MLP (default=1024)
    :param anchored: toggles weight anchoring of the MLP (default=False)
    :param l2_lambda: L2 loss scaling for the anchored MLP loss (default=1e-4)
    :param n_ensemble: number of MLPs in the ensemble (default=10)
    :param output_dim_mlp: dimensions of the output layer of the MLP (default=2)
    :param class_scaling_factor: Scales BCE loss for the '1' class, i.e. a factor of 2 would double the respective loss
        for the '1' class (default=None)
    :param variational_scale: The scale of the Gaussian of the encoder (default=1)
    :param kwargs: just here for compatability reasons.
    """
    def __init__(self, input_dim: int = 1024, latent_dim: int = 32, hidden_dim_vae: int = 1024, out_dim_vae: int = 1024,
                 beta: float = 0.001, n_layers_mlp: int = 2, hidden_dim_mlp: int = 1024, anchored: bool = True,
                 l2_lambda: float = 1e-4, n_ensemble: int = 10, output_dim_mlp: int = 2, class_scaling_factor: float = 1,
                 variational_scale: float = 1, device: str = None, **kwargs) -> None:
        super(JVAE, self).__init__()
        self.name = 'JVAE'
        self.device = device

        self.vae = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim_vae, out_dim=out_dim_vae,
                       beta=beta, class_scaling_factor=class_scaling_factor, variational_scale=variational_scale)
        self.prediction_head = Ensemble(input_dim=latent_dim, hidden_dim=hidden_dim_mlp, n_layers=n_layers_mlp,
                                        anchored=anchored, l2_lambda=l2_lambda, n_ensemble=n_ensemble,
                                        output_dim=output_dim_mlp)

    def forward(self, x: Tensor, y: Tensor = None, **kwargs):
        x_hat, z, sample_likelihood, loss = self.vae(x)
        y_logits_N_K_C, loss_mlp = self.prediction_head(z, y)

        if y is not None:
            loss = loss + loss_mlp  # TODO scale mlp loss?

        return y_logits_N_K_C, x_hat, z, sample_likelihood, loss

    def predict(self, dataset, pretrained_vae_path: str = None, batch_size: int = 128) -> Tensor:
        return _predict_jvae(self, dataset, pretrained_vae_path=pretrained_vae_path, batch_size=128)


def anchored_loss(model: MLP, x: Tensor, y: Tensor = None) -> Tensor:
    """ Compute anchored loss according to Pearce et al. (2018)

    :param model: MLP torch module
    :param x: model predictions
    :param y: target tensor (default = None)
    :return: loss or None (if y is None)
    """
    if y is None:
        return None

    loss_func = torch.nn.NLLLoss()
    loss = loss_func(x, y)

    if model.anchored:
        l2_loss = 0
        for p, p_a in zip(model.named_parameters(), model.named_buffers()):
            assert p_a[1].shape == p[1].shape
            l2_loss += (model.l2_lambda / len(y)) * torch.mul(p[1] - p_a[1], p[1] - p_a[1]).sum()
        loss = loss + l2_loss

    return loss


def _predict_mlp(model, dataset, batch_size: int = 128) -> Tensor:
    """ Get predictions from a dataloader

    :param model: torch module (e.g. MLP or Ensemble)
    :param dataset: dataset of the data to predict; jcm.datasets.MoleculeDataset
    :param batch_size: prediction batch size (default=128)

    :return: logits_N_K_C, vae latents, vae likelihoods
    """
    val_loader = DataLoader(dataset, sampler=None, shuffle=False, pin_memory=True, batch_size=batch_size)
    y_hats = []

    model.eval()
    for x, y in val_loader:
        # move to device
        x.to(model.device)

        if x.shape[0] == 1:
            x = x.squeeze(0).float()
        else:
            x = x.squeeze().float()

        # predict
        y_hat, loss = model(x)      # x_hat, z, sample_likelihood, loss
        y_hats.append(y_hat)

    model.train()

    return torch.cat(y_hats, 0)


def _predict_vae(model, dataset, batch_size: int = 128) -> (Tensor, Tensor, Tensor):
    """ Get predictions from a dataloader

    :param model: torch module (e.g. MLP or Ensemble)
    :param dataset: dataset of the data to predict; jcm.datasets.MoleculeDataset
    :param batch_size: prediction batch size (default=128)

    :return: logits_N_K_C, vae latents, vae likelihoods
    """
    val_loader = DataLoader(dataset, sampler=None, shuffle=False, pin_memory=True, batch_size=batch_size)
    y_hats = []
    zs = []
    sample_likelihoods = []

    model.eval()
    for x, y in val_loader:
        # move to device
        x.to(model.device)

        if x.shape[0] == 1:
            x = x.squeeze(0).float()
        else:
            x = x.squeeze().float()

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


def _predict_jvae(model, dataset, pretrained_vae_path: str = None, batch_size: int = 128) -> (Tensor, Tensor, Tensor):
    """ Get predictions from a dataloader

    :param model: torch module (e.g. MLP or Ensemble)
    :param dataset: dataset of the data to predict; jcm.datasets.MoleculeDataset
    :param pretrained_vae_path: path of the pretrained VAE, used to normalize likelihoods if supplied (default=None)
    :param batch_size: prediction batch size (default=128)

    :return: logits_N_K_C, x_reconstructions, vae latents, vae likelihoods
    """

    if pretrained_vae_path is not None:
        config = Config()
        config.set_hyperparameters(**VAE_PRETRAIN_HYPERPARAMETERS)

        pre_trained_vae = VAE(**config.hyperparameters)
        pre_trained_vae.load_state_dict(torch.load(pretrained_vae_path))
        pre_trained_vae.eval()

    val_loader = DataLoader(dataset, sampler=None, shuffle=False, pin_memory=True, batch_size=batch_size)

    y_hats = []
    x_hats = []
    zs = []
    sample_likelihoods = []
    pt_sample_likelihoods = []

    model.eval()
    for x, y in val_loader:
        # move to device
        x.to(model.device)

        if x.shape[0] == 1:
            x = x.squeeze(0).float()
        else:
            x = x.squeeze().float()

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
