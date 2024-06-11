"""
Contains all code for the variational encoder

Derek van Tilborg
Eindhoven University of Technology
June 2024
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class VariationalEncoder(nn.Module):
    """ A simple variational encoder. Takes a batch of vectors and compresses the input space to a smaller variational
    latent.

    :param var_input_dim: dimensions of the input layer (default=2048)
    :param z_size: dimensions of the latent/output layer (default=2048)
    :param variational_scale: The scale of the Gaussian of the encoder (default=1)
    """
    def __init__(self, var_input_dim: int = 2048, z_size: int = 128, variational_scale: float = 1, **kwargs):
        super(VariationalEncoder, self).__init__()
        self.name = 'VariationalEncoder'

        self.lin0_x = nn.Linear(var_input_dim, z_size)
        self.lin0_mu = nn.Linear(z_size, z_size)
        self.lin0_sigma = nn.Linear(z_size, z_size)

        self.N = torch.distributions.Normal(0, variational_scale)
        self.N.loc = self.N.loc
        self.N.scale = self.N.scale
        self.kl = 0

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.lin0_x(x))
        mu = self.lin0_mu(x)
        sigma = torch.exp(self.lin0_sigma(x))

        # reparameterization trick
        z = mu + sigma * self.N.sample(mu.shape).to('cuda' if torch.cuda.is_available() else 'cpu')  # TODO fix this
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()

        return z
