"""
Contains all code for the CNN encoder

Derek van Tilborg
Eindhoven University of Technology
June 2024
"""

from torch import nn, Tensor
import torch.nn.functional as F
from jcm.utils import calc_l_out


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

    def forward(self, x: Tensor) -> Tensor:

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
