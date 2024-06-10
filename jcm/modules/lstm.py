"""
Contains all LSTM code

Derek van Tilborg
Eindhoven University of Technology
June 2024
"""

import torch
from torch import nn as nn
from torch import Tensor
import torch.nn.functional as F
from jcm.utils import get_smiles_length_batch
from constants import VOCAB


class AutoregressiveLSTM(nn.Module):
    """ An autoregressive LSTM that takes integer-encoded SMILES strings and performs next token prediction.
    Negative Log Likelihood is calculated per molecule and per batch and is normalized per molecule length so that
    small molecules do not get an unfair advantage in loss over longer ones.

    :param hidden_size: size of the LSTM hidden layers (default=256)
    :param vocabulary_size: size of the vocab (default=36)
    :param num_layers: number of LSTM layers (default=2)
    :param embedding_dim: size of the SMILES embedding layer (default=128)
    :param dropout: dropout ratio, num_layers should be > 1 if dropout > 0 (default=0.2)
    :param device: device (default='cpu')
    :param ignore_index: index of the padding token (default=35, padding tokens must be ignored in this implementation)
    """

    def __init__(self, hidden_size: int = 256, vocabulary_size: int = 36, num_layers: int = 2, embedding_dim: int = 128,
                 ignore_index: int = 0, dropout: float = 0.2, device: str = 'cpu', **kwargs) -> None:
        super(AutoregressiveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.device = device
        self.ignore_index = ignore_index
        self.dropout = dropout

        self.loss_func = nn.NLLLoss(reduction='none', ignore_index=ignore_index)

        self.embedding_layer = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True,
                            num_layers=self.num_layers, dropout=dropout)
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocabulary_size)

    def forward(self, x: Tensor, *args) -> (Tensor, Tensor, Tensor, Tensor):
        """ Perform next-token autoregression on a batch of SMILES strings

        :param x: :math:`(N, S)`, integer encoded SMILES strings where S is sequence length, as .long()
        :param args: redundant param that is kept for compatability
        :return:  predicted token probability, molecule embedding, molecule loss, batch loss
        """

        # find the position of the first occuring padding token, which is the length of the SMILES
        length_of_smiles = get_smiles_length_batch(x)
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # turn indexed encoding into embeddings
        embedding = self.embedding_layer(x)

        # init an empty hidden and cell state for the first token
        hidden_state, cell_state = init_lstm_hidden(num_layers=self.num_layers, batch_size=batch_size,
                                                    hidden_size=self.hidden_size, device=self.device)

        loss, probs = [], []
        for t_i in range(seq_len - 1):  # loop over all tokens in the sequence

            # Get the current and next token in the sequence
            x_i = embedding[:, t_i, :].unsqueeze(1)  # (batch_size, 1, vocab_size)
            next_token = x[:, t_i + 1]  # (batch_size, vocab_size)

            # predict the next token in the sequence
            x_hat, (hidden_state, cell_state) = self.lstm(x_i, (hidden_state, cell_state))
            logits = F.relu(self.fc(x_hat))  # (batch_size, 1, vocab_size)
            x_probs = F.softmax(logits, dim=-1).squeeze()  # (batch_size, vocab_size)

            # Compute loss
            loss_i = self.loss_func(logits.squeeze(), next_token)  # (batch_size)

            probs.append(x_probs)
            loss.append(loss_i)

        # stack the token-wise losses and the predicted token probabilities
        loss = torch.stack(loss, 1)
        probs = torch.stack(probs, 1)

        # Sum up the losses (padding should be ignored so this doesn't affect the sum) and normalize by SMILES length
        # This is a mean Loss that is not biased by SMILES length. If you don't do this, longer SMILES get penalized.
        sample_loss = torch.sum(loss, 1) / length_of_smiles

        return probs, embedding, sample_loss, torch.mean(sample_loss)


class SMILESTokenLoss(torch.nn.Module):
    """ Calculates the Negative Log Likelihood for a batch of predicted SMILES token given logits and target values.

    If SMILES lenghts are provided in the forward, token losses are normalized by the length of the corresponding
    SMILES string

    Args:
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient.

    Shape:
        - Input: :math:`(N, C)`, where `C = number of classes`, in .float()
        - Target: :math:`(N)`, where each value is :math:`0 \leq target[i] \leq C-1`, in .long()
        - Output: :math:`(N)`.

    Examples::

        >>> loss_function = SMILESTokenLoss()
        >>> # For a batch size of 3 and a vocab size of 36
        >>> smiles_lengths = torch.tensor([45, 53, 27])
        >>> logits = torch.randn(3, 36, requires_grad=True)
        >>> target = torch.tensor([1, 0, 4])
        >>> loss = loss_function(logits, target, length_norm=smiles_lengths)
        >>> loss.backward()

     """

    def __init__(self, ignore_index=0):
        super(SMILESTokenLoss, self).__init__()
        self.loss_func = nn.NLLLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, logits: Tensor, target: Tensor, length_norm: Tensor = None):
        """ Compute token loss

        :param logits: :math:`(N, C)`, token logits
        :param target: :math:`(N)`, target tokens
        :param length_norm: :math:`(N)`, Tensor of SMILES lengths to normalize each loss (default=None)
        :return: :math:`(N)`, loss for each token with shape
        """

        log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, vocab_size)
        token_loss = self.loss_func(log_probs, target)  # (batch_size)

        if length_norm is not None:
            token_loss = token_loss / length_norm

        return token_loss


def init_lstm_hidden(num_layers, batch_size, hidden_size, device):
    # Initialize hidden and cell states with zeros

    h_0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
    c_0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)

    return h_0, c_0