"""
Contains all code for the CNN encoder

Derek van Tilborg
Eindhoven University of Technology
June 2024
"""

from torch import nn, Tensor
from torch.nn import functional as F


class CnnEncoder(nn.Module):
    """ Encode a one-hot encoded SMILES string with a CNN. Uses Max Pooling and flattens conv layer at the end

    :param vocabulary_size: vocab size (default=36)
    :param seq_length: sequence length of SMILES strings (default=102)
    :param cnn_out_hidden: dimension of the CNN token embedding size (default=256)
    :param cnn_kernel_size: CNN kernel_size (default=8)
    :param cnn_n_layers: number of layers in the CNN (default=3)
    :param cnn_stride: stride (default=1)
    """

    def __init__(self, token_embedding_dim: int = 128, seq_length: int = 102, cnn_out_hidden: int = 256,
                 cnn_kernel_size: int = 8, cnn_stride: int = 1, cnn_n_layers: int = 3, **kwargs):
        super().__init__()
        self.n_layers = cnn_n_layers
        assert cnn_n_layers <= 3, f"The CNN can have between 1 and 3 layers, not: cnn_n_layers={cnn_n_layers}."

        self.pool = nn.MaxPool1d(kernel_size=cnn_kernel_size, stride=cnn_stride)
        if cnn_n_layers == 1:
            self.cnn0 = nn.Conv1d(token_embedding_dim, cnn_out_hidden, kernel_size=cnn_kernel_size, stride=cnn_stride)
            self.l_out = calc_l_out(seq_length, self.cnn0, self.pool)
        if cnn_n_layers == 2:
            self.cnn0 = nn.Conv1d(token_embedding_dim, 128, kernel_size=cnn_kernel_size, stride=cnn_stride)
            self.cnn1 = nn.Conv1d(128, cnn_out_hidden, kernel_size=cnn_kernel_size, stride=cnn_stride)
            self.l_out = calc_l_out(seq_length, self.cnn0, self.pool, self.cnn1, self.pool)
        if cnn_n_layers == 3:
            self.cnn0 = nn.Conv1d(token_embedding_dim, 64, kernel_size=cnn_kernel_size, stride=cnn_stride)
            self.cnn1 = nn.Conv1d(64, 128, kernel_size=cnn_kernel_size, stride=cnn_stride)
            self.cnn2 = nn.Conv1d(128, cnn_out_hidden, kernel_size=cnn_kernel_size, stride=cnn_stride)
            self.l_out = calc_l_out(seq_length, self.cnn0, self.pool, self.cnn1, self.pool, self.cnn2, self.pool)

        self.out_dim = int(cnn_out_hidden * self.l_out)

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


def calc_l_out(l: int, *models) -> int:
    """ Calculate the sequence length of a series of conv/pool torch models from a starting sequence length

    :param l: sequence_length
    :param models: pytorch models
    :return: sequence length of the final model
    """
    def cnn_out_l_size(cnn, l):
        if type(cnn.padding) is int:
            return ((l + (2 * cnn.padding) - (cnn.dilation * (cnn.kernel_size - 1)) - 1) / cnn.stride) + 1
        else:
            return ((l + (2 * cnn.padding[0]) - (cnn.dilation[0] * (cnn.kernel_size[0] - 1)) - 1) / cnn.stride[0]) + 1

    for m in models:
        l = cnn_out_l_size(m, l)
    return l
