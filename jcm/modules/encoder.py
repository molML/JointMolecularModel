"""
Contains all code for the CNN encoder

Derek van Tilborg
Eindhoven University of Technology
June 2024
"""

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from jcm.modules.variational import VariationalEncoder


class Encoder(nn.Module):
    """ SMILES -> CNN -> z """

    def __init__(self, token_embedding_dim: int = 128, vocabulary_size: int = 36, variational: bool = False,
                 seq_length: int = 102, cnn_out_hidden: int = 256, cnn_kernel_size: int = 8, cnn_stride: int = 1,
                 cnn_n_layers: int = 3, beta: float = 1., z_size: int = 128, sigma_prior: float = 0.1,
                 cnn_dropout: float = 0.1, **kwargs):
        super(Encoder, self).__init__()

        self.register_buffer('beta', torch.tensor(beta))
        self.token_embedding_dim = token_embedding_dim
        self.vocabulary_size = vocabulary_size
        self.variational = variational
        self.seq_length = seq_length
        self.cnn_out_hidden = cnn_out_hidden
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_stride = cnn_stride
        self.cnn_n_layers = cnn_n_layers
        self.cnn_dropout = cnn_dropout
        self.z_size = z_size
        self.sigma_prior = sigma_prior

        self.embedding_layer = nn.Embedding(num_embeddings=vocabulary_size,
                                            embedding_dim=token_embedding_dim)

        self.cnn = CNN(token_embedding_dim=token_embedding_dim,
                       seq_length=seq_length,
                       cnn_out_hidden=cnn_out_hidden,
                       cnn_kernel_size=cnn_kernel_size,
                       cnn_stride=cnn_stride,
                       cnn_n_layers=cnn_n_layers,
                       cnn_dropout=cnn_dropout)

        if self.variational:
            self.z_layer = VariationalEncoder(var_input_dim=self.cnn.out_dim,
                                              z_size=z_size,
                                              sigma_prior=sigma_prior)
        else:
            self.z_layer = nn.Linear(self.cnn.out_dim, self.z_size)

        self.kl_loss = None

    def forward(self, x: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        """ Encode a batch of molecule

        :param x: :math:`(N, C)`, batch of integer encoded molecules
        :return: z
        """

        # Embed the integer encoded molecules with the same embedding layer that is used later in the rnn
        # We transpose it from (batch size x sequence length x embedding) to (batch size x embedding x sequence length)
        # so the embedding is the channel instead of the sequence length
        embedding = self.embedding_layer(x).transpose(1, 2)

        # Encode the molecule into a latent vector z
        z = self.z_layer(self.cnn(embedding))

        if self.variational:
            self.kl_loss = self.beta * self.z_layer.kl  # / x.shape[0]

        return z


class CNN(nn.Module):
    """ Encode a one-hot encoded SMILES string with a CNN. Uses Max Pooling and flattens conv layer at the end

    :param vocabulary_size: vocab size (default=36)
    :param seq_length: sequence length of SMILES strings (default=102)
    :param cnn_out_hidden: dimension of the CNN token embedding size (default=256)
    :param cnn_kernel_size: CNN kernel_size (default=8)
    :param cnn_n_layers: number of layers in the CNN (default=3)
    :param cnn_stride: stride (default=1)
    """

    def __init__(self, token_embedding_dim: int = 128, seq_length: int = 102, cnn_out_hidden: int = 256,
                 cnn_kernel_size: int = 8, cnn_stride: int = 1, cnn_n_layers: int = 3, cnn_dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.n_layers = cnn_n_layers
        assert cnn_n_layers <= 3, f"The CNN can have between 1 and 3 layers, not: cnn_n_layers={cnn_n_layers}."

        self.pool = nn.MaxPool1d(kernel_size=cnn_kernel_size, stride=cnn_stride)
        self.dropout = nn.Dropout(p=cnn_dropout)
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
        x = self.dropout(x)

        if self.n_layers == 2:
            x = F.relu(self.cnn1(x))
            x = self.pool(x)
            x = self.dropout(x)

        if self.n_layers == 3:
            x = F.relu(self.cnn1(x))
            x = self.pool(x)
            x = self.dropout(x)

            x = F.relu(self.cnn2(x))
            x = self.pool(x)
            x = self.dropout(x)

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
