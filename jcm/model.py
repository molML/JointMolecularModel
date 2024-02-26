
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
                 seed: int = 42, anchored: bool = False, l2_lambda: float = 1e-4):
        super().__init__()
        torch.manual_seed(seed)
        self.l2_lambda = l2_lambda
        self.anchored = anchored

        self.fc = torch.nn.ModuleList()
        for i in range(n_layers):
            if anchored:
                self.fc.append(AnchoredLinear(input_dim if i == 0 else hidden_dim, hidden_dim))
            else:
                self.fc.append(torch.nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
        if anchored:
            self.out = AnchoredLinear(hidden_dim, output_dim)
        else:
            self.out = torch.nn.Linear(hidden_dim, output_dim)

    def reset_parameters(self):
        for lin in self.fc:
            lin.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor, y: Tensor = None) -> (Tensor, Tensor):
        for lin in self.fc:
            x = F.relu(lin(x))
        x = self.out(x)
        x = F.log_softmax(x, 1)

        loss_func = torch.nn.NLLLoss()
        loss = None
        if y is not None:
            loss = loss_func(x, y)
            loss_original = loss

            if self.anchored:
                l2_loss = 0
                for p, p_a in zip(self.named_parameters(), self.named_buffers()):
                    assert p_a[1].shape == p[1].shape
                    l2_loss += (self.l2_lambda / len(y)) * torch.mul(p[1] - p_a[1], p[1] - p_a[1]).sum()
                    # Add anchored loss to regular loss according to Pearce et al. (2018)
                    loss = loss + l2_loss
                print(loss_original, l2_loss)

        return x, loss


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
        self.register_buffer('anchor_weight', self.weight.clone().detach())
        self.register_buffer('anchor_bias', self.bias.clone().detach())

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)


class Ensemble(nn.Module):

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 1024, output_dim: int = 2, n_layers: int = 2,
                 anchored: bool = False, l2_lambda: float = 1e-4, n_ensemble: int = 10):
        super().__init__()
        self.mlps = nn.ModuleList()
        for i in range(n_ensemble):
            self.mlps.append(MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers,
                                 seed=i, anchored=anchored, l2_lambda=l2_lambda))

    def forward(self, x: Tensor, y: Tensor = None):

        loss = Tensor([0])
        y_hats = []
        for mlp_i in self.mlps:
            y_hat_i, loss_i = mlp_i(x, y)
            y_hats.append(y_hat_i)
            loss += loss_i

        loss = None if y is None else loss/len(self.mlps)
        logits_N_K_C = torch.stack(y_hats)

        return logits_N_K_C, loss


def logits_to_pred(logits_N_K_C: Tensor, return_prob: bool = True, return_uncertainty: bool = True) -> (Tensor, Tensor):
    """ Get the probabilities/class vector and sample uncertainty from the logits """

    mean_probs_N_C = torch.mean(torch.exp(logits_N_K_C), dim=1)
    uncertainty = mean_sample_entropy(logits_N_K_C)

    if return_prob:
        y_hat = mean_probs_N_C
    else:
        y_hat = torch.argmax(mean_probs_N_C, dim=1)

    if return_uncertainty:
        return y_hat, uncertainty
    else:
        return y_hat


def logit_mean(logits_N_K_C: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """ Logit mean with the logsumexp trick - Kirch et al., 2019, NeurIPS """

    return torch.logsumexp(logits_N_K_C, dim=dim, keepdim=keepdim) - math.log(logits_N_K_C.shape[dim])


def entropy(logits_N_K_C: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """Calculates the Shannon Entropy """

    return -torch.sum((torch.exp(logits_N_K_C) * logits_N_K_C).double(), dim=dim, keepdim=keepdim)


def mean_sample_entropy(logits_N_K_C: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    """Calculates the mean entropy for each sample given multiple ensemble predictions - Kirch et al., 2019, NeurIPS"""

    sample_entropies_N_K = entropy(logits_N_K_C, dim=dim, keepdim=keepdim)
    entropy_mean_N = torch.mean(sample_entropies_N_K, dim=1)

    return entropy_mean_N


def mutual_information(logits_N_K_C: Tensor) -> Tensor:
    """ Calculates the Mutual Information - Kirch et al., 2019, NeurIPS """

    # this term represents the entropy of the model prediction (high when uncertain)
    entropy_mean_N = mean_sample_entropy(logits_N_K_C)

    # This term is the expectation of the entropy of the model prediction for each draw of model parameters
    mean_entropy_N = entropy(logit_mean(logits_N_K_C, dim=1), dim=-1)

    I = mean_entropy_N - entropy_mean_N

    return I


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
                 beta: float = 0.001, class_scaling_factor: float = 1, **kwargs):
        super(VAE, self).__init__()

        self.register_buffer('beta', torch.tensor(beta))
        self.register_buffer('class_scaling_factor', torch.tensor(class_scaling_factor))
        self.encoder = VariationalEncoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = Decoder(input_dim=latent_dim, hidden_dim=hidden_dim, out_dim=out_dim)

    def forward(self, x: Tensor, y: Tensor = None):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        # compute losses
        loss_reconstruction, sample_likelihood = BCE_per_sample(x_hat, x, self.class_scaling_factor)
        loss_kl = self.encoder.kl / x.shape[0]
        loss = loss_reconstruction + self.beta * loss_kl

        return x_hat, z, sample_likelihood, loss


class JVAE(nn.Module):

    def __init__(self, input_dim: int = 1024, latent_dim: int = 32, hidden_dim_vae: int = 1024, out_dim_vae: int = 1024,
                 beta: float = 0.001, n_layers_mlp: int = 2, hidden_dim_mlp: int = 1024, anchored: bool = True,
                 l2_lambda: float = 1e-4, n_ensemble: int = 10, output_dim_mlp: int = 2, class_scaling_factor: float = 1,
                 **kwargs):
        super(JVAE, self).__init__()

        self.vae = VAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim_vae, out_dim=out_dim_vae,
                       beta=beta, class_scaling_factor=class_scaling_factor)
        self.prediction_head = Ensemble(input_dim=latent_dim, hidden_dim=hidden_dim_mlp, n_layers=n_layers_mlp,
                                        anchored=anchored, l2_lambda=l2_lambda, n_ensemble=n_ensemble,
                                        output_dim=output_dim_mlp)

    def forward(self, x: Tensor, y: Tensor = None, **kwargs):
        x_hat, z, sample_likelihood, loss_vae = self.vae(x)
        y_logits_N_K_C, loss_mlp = self.prediction_head(z, y)

        loss = loss_vae + loss_mlp  #TODO scale mlp loss?

        return y_logits_N_K_C, z, sample_likelihood, loss

#
# x = torch.rand((32, 1024))
# y = torch.randint(0, 2, [32])
# f = Ensemble(anchored=True)
#
# # 0.6884
# f = MLP(anchored=True)
# f.reset_parameters()
#
# logits, loss = f(x, y)
#
# loss.shape
#
# model = JVAE(latent_dim=128)
# model.vae.load_state_dict(torch.load('/Users/derekvantilborg/Dropbox/PycharmProjects/JointChemicalModel/results/chembl_vae/pretrained_vae.pt'))
#
# model.vae._buffers
#
# y_logits_N_K_C, z, sample_likelihood, loss = model(x, y)
# z.shape

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

