
import math
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.nn.parameter import Parameter
from torch.nn import init
from jcm.config import Config
from jcm.utils import to_binary


class MLP(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 1024, output_dim: int = 2, n_layers: int = 2,
                 seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)

        self.fc = torch.nn.ModuleList()
        for i in range(n_layers):
            self.fc.append(torch.nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
        self.out = torch.nn.Linear(hidden_dim, output_dim)

    def reset_parameters(self):
        for lin in self.fc:
            lin.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for lin in self.fc:
            x = F.relu(lin(x))

        x = self.out(x)

        return x


class AnchoredLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # store the init weight/bias as a buffer
        self.register_buffer('anchor_weight', self.weight)
        self.register_buffer('anchor_bias', self.bias)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)


# x = torch.rand((32, 1024))
# f = AnchoredLinear(1024, 1024)
#
# f(x)
#
# for b in f._buffers:
#     print(b.shape)
#
# f._buffers



# class Ensemble(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#     def reset_parameters(self):
#         pass
#
#     def forward(self):
#         pass


class Decoder(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 1024, out_dim: int = 1024, n_layers: int = 1):
        super(Decoder, self).__init__()

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
        # z = F.log_softmax(self.lin2(z), -1)

        return z


class VariationalEncoder(nn.Module):
    def __init__(self, input_dim: int = 1024, latent_dim: int = 128):
        super(VariationalEncoder, self).__init__()

        self.lin0_x = nn.Linear(input_dim, latent_dim)
        self.lin0_mu = nn.Linear(latent_dim, latent_dim)
        self.lin0_sigma = nn.Linear(latent_dim, latent_dim)

        self.N = torch.distributions.Normal(0, 1)
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


def scale_BCE(loss: Tensor, y: Tensor, factor: float = 1) -> Tensor:
    """ Scales BCE loss for the 1 class, i.e., a scaling factor of 2 would weight the '1' class twice as heavy as the
    '0' class. Requires BCE loss with reduction='none'

    :param loss: BCELoss with reduction='none'
    :param y: labels, in the same shape as the loss (which happens with reduction='none')
    :param factor: scaling factor (default=1)
    :return: scaled loss
    """
    scaling_tensor = torch.ones(loss.shape)
    scaling_tensor[y == 1] = factor

    return loss * scaling_tensor


def BCE_per_sample(y_hat: Tensor, y: Tensor, class_scaling_factor: float = None) -> (Tensor, Tensor):
    """ Computes the BCE loss and also returns the summed BCE per individual samples

    :param y_hat: predictions [batch_size, binary bits]
    :param y: labels [batch_size, binary bits]
    :param class_scaling_factor: Scales BCE loss for the '1' class, i.e. a factor of 2 would double the respective loss
        for the '1' class (default=None)
    :return: overall batch BCE loss, BCE per sample
    """

    loss_fn = nn.BCELoss(reduction='none')
    loss = loss_fn(y_hat, y.float())
    sample_loss = torch.mean(loss, 1)

    if class_scaling_factor is not None:
        loss = scale_BCE(loss, y, factor=class_scaling_factor)

    return torch.mean(loss), sample_loss


class VAE(nn.Module):
    def __init__(self, input_dim: int = 1024, latent_dim: int = 32, hidden_dim: int = 1024, out_dim: int = 1024,
                 beta: float = 0.001, class_scaling_factor: float = None, **kwargs):
        super(VAE, self).__init__()

        self.register_buffer('beta', torch.tensor(beta))
        self.register_buffer('class_scaling_factor', torch.tensor(class_scaling_factor))
        self.encoder = VariationalEncoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = Decoder(input_dim=latent_dim, hidden_dim=hidden_dim, out_dim=out_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        # compute losses
        loss_reconstruction, sample_likelihood = BCE_per_sample(x_hat, x, self.class_scaling_factor)
        loss_kl = self.encoder.kl / x.shape[0]
        loss = loss_reconstruction + self.beta * loss_kl

        return x_hat, z, sample_likelihood, loss


class JVAE(nn.Module):

    def __init__(self, input_dim: int = 1024, latent_dim: int = 32, hidden_dim: int = 1024, out_dim: int = 1024,
                 beta: float = 0.001):
        super(JVAE, self).__init__()

        self.vae = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, out_dim=out_dim, beta=beta)
        self.prediction_head = ...

    def forward(self, x, **kwargs):
        x_hat, z, sample_likelihood, loss_vae = self.vae(x)
        y_hat = self.prediction_head(z)

        loss_mlp = ...
        loss = loss_vae + loss_mlp + ...

        return y_hat, z, sample_likelihood, loss


####
#
# m = nn.Sigmoid()
# loss = nn.BCELoss(reduction='none')
# input = torch.randn(3, 2, requires_grad=True)
# target = torch.rand(3, 2, requires_grad=False)
# output = loss(m(input), target)
# output.backward()
#
# import pandas as pd
# from dataprep.utils import smiles_to_mols
# from dataprep.descriptors import mols_to_ecfp, mols_to_maccs
#
# df = pd.read_csv('data/moleculeace/CHEMBL234_Ki.csv')
# df_train = df[df['split'] == 'train']
# df_test = df[df['split'] == 'test']
#
# x_train = torch.tensor(mols_to_ecfp(smiles_to_mols(df_train.smiles), to_array=True))
# y_train = binary_tensor = to_binary(torch.tensor(df_train['exp_mean [nM]'].tolist()))
#
# x_test = torch.tensor(mols_to_ecfp(smiles_to_mols(df_test.smiles), to_array=True))
# y_test = binary_tensor = to_binary(torch.tensor(df_test['exp_mean [nM]'].tolist()))
#
#
# train_loader = DataLoader(x_train, batch_size=128)
# test_loader = DataLoader(x_test, batch_size=128)
# for x in test_loader:
#     break
#     x.shape
#
# model = VAE(input_dim=1024, latent_dim=64, out_dim=1024, beta=0.01, class_scaling_factor=2)
#
# from jcm.trainer import Trainer
# from sklearn.metrics import balanced_accuracy_score, f1_score
#
# config = Config(max_iters=20000)
# T = Trainer(config, model, x_train)
# T.run()
# #
# #
# x_hat, z, sample_likelihood, loss = model(x.float())
#
#
# y = x[1]
# y_hat = to_binary(x_hat[1])
#
#
#
#
# c = ClassificationMetrics(y, y_hat)
# c
#
#
# #
# # # mean acc for batch
# sum([balanced_accuracy_score(x[i], to_binary(x_hat[i])) for i in range(len(x_hat))])/len(x_hat)
# #
#
# x[1] == to_binary(x_hat[1])
# sum(x[1])
# sum(to_binary(x_hat[1]))
#
#
# to_binary(x_hat[1])[x[1] == 1]
#
#

