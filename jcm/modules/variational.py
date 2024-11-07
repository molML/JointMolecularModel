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
    :param sigma_prior: The scale of the Gaussian of the encoder (default=1)
    """
    def __init__(self, var_input_dim: int = 2048, z_size: int = 128, sigma_prior: float = 1., device: str = 'cpu',
                 **kwargs):
        super(VariationalEncoder, self).__init__()
        self.name = 'VariationalEncoder'
        self.register_buffer('sigma_prior', torch.tensor(sigma_prior))
        self.device = device

        self.lin0_x = nn.Linear(var_input_dim, z_size)
        self.lin0_mu = nn.Linear(z_size, z_size)
        self.lin0_sigma = nn.Linear(z_size, z_size)

        self.N = torch.distributions.Normal(0, self.sigma_prior)
        self.N.loc = self.N.loc
        self.N.scale = self.N.scale
        self.kl = 0

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.lin0_x(x))
        mu = self.lin0_mu(x)
        sigma = torch.exp(self.lin0_sigma(x))

        # reparameterization trick
        z = mu + sigma * self.N.sample(mu.shape).to(self.device)
        self.kl = 0.5 * ((sigma**2 / self.sigma_prior**2) + (mu**2 / self.sigma_prior**2) - torch.log(sigma**2 / self.sigma_prior**2) - 1).sum(dim=1)

        return z
