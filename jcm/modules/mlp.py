"""
Contains all code for MLPs including Anchored (Pearce et al. (2018)) ones

Derek van Tilborg
Eindhoven University of Technology
June 2024
"""

import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.utils.data.dataloader import DataLoader
from jcm.utils import single_batchitem_fix


class MLP(nn.Module):
    """ Multi-Layer Perceptron with weight anchoring according to Pearce et al. (2018)

    :param mlp_input_dim: input layer dimension (default=2048)
    :param hmlp_idden_dim: hidden layer(s) dimension (default=2048)
    :param mlp_output_dim: output layer dimension (default=2)
    :param mlp_n_layers: number of layers (including the input layer, not including the output layer, default=2)
    :param seed: random seed (default=42)
    :param mlp_anchored: toggles weight anchoring (default=False)
    :param mlp_l2_lambda: L2 loss scaling for the anchored loss (default=1e-4)
    :param device: 'cpu' or 'cuda' (default=None)
    """
    def __init__(self, mlp_input_dim: int = 2048, mlp_hidden_dim: int = 2048, mlp_output_dim: int = 2,
                 mlp_n_layers: int = 2, seed: int = 42, mlp_anchored: bool = False, mlp_l2_lambda: float = 1e-4,
                 device: str = None, **kwargs) -> None:
        super().__init__()
        torch.manual_seed(seed)

        self.l2_lambda = mlp_l2_lambda
        self.anchored = mlp_anchored
        self.name = 'MLP'
        self.device = device

        self.fc = torch.nn.ModuleList()
        for i in range(mlp_n_layers):
            self.fc.append(AnchoredLinear(mlp_input_dim if i == 0 else mlp_hidden_dim, mlp_hidden_dim, device=self.device))
        self.out = AnchoredLinear(mlp_hidden_dim, mlp_output_dim, device=self.device)

    def reset_parameters(self):
        for lin in self.fc:
            lin.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x: Tensor, y: Tensor = None) -> (Tensor, Tensor, Tensor):
        """ Predict target classes from a molecular vector

        :param x: :math:`(N, H)`, input vector
        :param y: :math:`(N, C)`, target vector
        :return: log probabilties, molecule loss, loss
        """

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

    :param in_features: :math:`(N, H_in)`, size of each input sample
    :param out_features: :math:`(N, H_out)`, size of each output sample
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


# class EnsembleFrame(nn.Module):
#     """ An ensemble of (anchored) MLPs, used for uncertainty estimation. Outputs logits_N_K_C (n_ensemble, batch_size,
#     classes) and a (regularized) NLL loss
#
#     :param input_dim: dimensions of the input layer (default=2048)
#     :param hidden_dim: dimensions of the hidden layer(s) (default=2048)
#     :param output_dim: dimensions of the output layer (default=2)
#     :param n_layers: number of layers (including the input layer, not including the output layer, default=2)
#     :param anchored: toggles the use of anchored loss regularization, Pearce et al. (2018) (default=True)
#     :param l2_lambda: L2 loss scaling for the anchored loss (default=1e-4)
#     :param n_ensemble: number of models in the ensemble (default=10)
#     :param device: 'cpu' or 'cuda' (default=None)
#     """
#     def __init__(self, input_dim: int = 2048, latent_dim: int = 64, hidden_dim: int = 2048, output_dim: int = 2,
#                  n_layers: int = 2, anchored: bool = True, l2_lambda: float = 1e-4, n_ensemble: int = 10,
#                  device: str = None, **kwargs) -> None:
#         super().__init__()
#         self.name = 'EnsembleFrame'
#         self.device = device
#
#         # latent_dim
#         self.compress = nn.Linear(input_dim, latent_dim)
#         self.ensemble = Ensemble(input_dim=latent_dim, hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers,
#                                  anchored=anchored, l2_lambda=l2_lambda, n_ensemble=n_ensemble, device=device)
#
#     def forward(self, x: Tensor, y: Tensor = None):
#
#         x = F.relu(self.compress(x))
#         logits_N_K_C, loss = self.ensemble(x, y)
#
#         return logits_N_K_C, loss
#
#     def predict(self, dataset, batch_size: int = 128) -> Tensor:
#         return _predict_mlp(self, dataset, batch_size)


class Ensemble(nn.Module):
    """ An ensemble of (anchored) MLPs, used for uncertainty estimation. Outputs logits_N_K_C (n_ensemble, batch_size,
    classes) and a (regularized) NLL loss

    :param mlp_input_dim: dimensions of the input layer (default=2048)
    :param mlp_hidden_dim: dimensions of the hidden layer(s) (default=2048)
    :param mlp_output_dim: dimensions of the output layer (default=2)
    :param mlp_n_layers: number of layers (including the input layer, not including the output layer, default=2)
    :param mlp_anchored: toggles the use of anchored loss regularization, Pearce et al. (2018) (default=True)
    :param mlp_l2_lambda: L2 loss scaling for the anchored loss (default=1e-4)
    :param mlp_n_ensemble: number of models in the ensemble (default=10)
    :param device: 'cpu' or 'cuda' (default=None)
    """
    def __init__(self, mlp_input_dim: int = 2048, mlp_hidden_dim: int = 2048, mlp_output_dim: int = 2,
                 mlp_n_layers: int = 2, mlp_anchored: bool = True, mlp_l2_lambda: float = 1e-4,
                 mlp_n_ensemble: int = 10, device: str = None, **kwargs) -> None:
        super().__init__()
        self.name = 'Ensemble'
        self.device = device

        self.mlps = nn.ModuleList()
        for i in range(mlp_n_ensemble):
            self.mlps.append(MLP(mlp_input_dim=mlp_input_dim, mlp_hidden_dim=mlp_hidden_dim,
                                 mlp_output_dim=mlp_output_dim, mlp_n_layers=mlp_n_layers,
                                 seed=i, mlp_anchored=mlp_anchored, mlp_l2_lambda=mlp_l2_lambda, device=device))

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


def anchored_loss(model: MLP, x: Tensor, y: Tensor = None) -> (Tensor, Tensor):
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
