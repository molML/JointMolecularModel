"""
Contains all LSTM code

Derek van Tilborg
Eindhoven University of Technology
June 2024
"""

import torch
from torch import nn as nn
from torch import Tensor
from torch.nn import functional as F
from jcm.utils import get_smiles_length_batch
from constants import VOCAB


class AutoregressiveLSTM(nn.Module):
    """ An autoregressive LSTM that takes integer-encoded SMILES strings and performs next token prediction.
    Negative Log Likelihood is calculated per molecule and per batch and is normalized per molecule length so that
    small molecules do not get an unfair advantage in loss over longer ones.

    :param lstm_hidden_size: size of the LSTM hidden layers (default=256)
    :param vocabulary_size: size of the vocab (default=36)
    :param lstm_num_layers: number of LSTM layers (default=2)
    :param lstm_embedding_dim: size of the SMILES embedding layer (default=128)
    :param lstm_dropout: dropout ratio, num_layers should be > 1 if dropout > 0 (default=0.2)
    :param device: device (default='cpu')
    :param ignore_index: index of the padding token (default=35, padding tokens must be ignored in this implementation)
    """

    def __init__(self, lstm_hidden_size: int = 256, vocabulary_size: int = 36, lstm_num_layers: int = 2,
                 token_embedding_dim: int = 128, ignore_index: int = 0, lstm_dropout: float = 0.2, device: str = 'cpu',
                 **kwargs) -> None:
        super(AutoregressiveLSTM, self).__init__()
        self.hidden_size = lstm_hidden_size
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = token_embedding_dim
        self.num_layers = lstm_num_layers
        self.device = device
        self.ignore_index = ignore_index
        self.dropout = lstm_dropout

        self.loss_func = SMILESTokenLoss(ignore_index=ignore_index)

        self.embedding_layer = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=token_embedding_dim)
        self.lstm = nn.LSTM(input_size=token_embedding_dim, hidden_size=lstm_hidden_size, batch_first=True,
                            num_layers=self.num_layers, dropout=lstm_dropout)
        self.fc = nn.Linear(in_features=lstm_hidden_size, out_features=vocabulary_size)

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

        token_losses, token_probs = [], []
        for t_i in range(seq_len - 1):  # loop over all tokens in the sequence

            # Get the current and next token in the sequence
            x_i = embedding[:, t_i, :].unsqueeze(1)  # (batch_size, 1, vocab_size)
            next_token = x[:, t_i + 1]  # (batch_size, vocab_size)

            # predict the next token in the sequence
            x_hat, (hidden_state, cell_state) = self.lstm(x_i, (hidden_state, cell_state))
            logits = F.relu(self.fc(x_hat))  # (batch_size, 1, vocab_size)
            probs = F.softmax(logits, dim=-1).squeeze()  # (batch_size, vocab_size)

            # Compute loss
            token_loss = self.loss_func(logits.squeeze(), next_token, length_of_smiles)

            token_probs.append(probs)
            token_losses.append(token_loss)

        # stack the token-wise losses and the predicted token probabilities
        token_losses = torch.stack(token_losses, 1)
        token_probs_N_S_C = torch.stack(token_probs, 1)

        # Sum up the token losses to get molecule-wise loss and average out over them to get the overall loss
        molecule_loss = torch.sum(token_losses, 1)
        loss = torch.mean(molecule_loss)

        return token_probs_N_S_C, molecule_loss, loss


class DecoderLSTM(nn.Module):

    def __init__(self, lstm_hidden_size: int = 256, vocabulary_size: int = 36, lstm_num_layers: int = 2,
                 token_embedding_dim: int = 128, z_size: int = 128, ignore_index: int = 0, lstm_dropout: float = 0.2,
                 device: str = 'cpu', **kwargs) -> None:
        super(DecoderLSTM, self).__init__()

        self.hidden_size = lstm_hidden_size
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = token_embedding_dim
        self.num_layers = lstm_num_layers
        self.device = device
        self.ignore_index = ignore_index
        self.dropout = lstm_dropout

        self.loss_func = SMILESTokenLoss(ignore_index=ignore_index)

        self.lstm = nn.LSTM(input_size=token_embedding_dim, hidden_size=lstm_hidden_size, batch_first=True,
                            num_layers=lstm_num_layers, dropout=lstm_dropout)
        self.z_transform = nn.Linear(in_features=z_size, out_features=lstm_hidden_size * lstm_num_layers)
        self.lin_lstm_to_token = nn.Linear(in_features=lstm_hidden_size, out_features=vocabulary_size)
        self.embedding_layer = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=token_embedding_dim)

    def condition_lstm(self, z: Tensor) -> (Tensor, Tensor):
        """ Condition the initial hidden state of the lstm with a latent vector z

        :param z: :math:`(N, Z)`, batch of latent molecule representations
        :return: :math:`(L, N, H), (L, N, H)`, hidden state & cell state, where L is num_layers, H is LSTM hidden size
        """

        batch_size = z.shape[0]
        # transform z to lstm_hidden_size * lstm_num_layers
        z = F.relu(self.z_transform(z))

        # reshape z into the lstm hidden state so it's distributed over the num_layers. This makes sure that for each
        # item in the batch, it's split into num_layers chunks, with shape (num_layers, batch_size, hidden_size) so
        # that the conditioned information is still matched for each item in the batch
        h_0 = z.reshape(batch_size, self.num_layers, self.hidden_size).transpose(1, 0).contiguous()
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)

        return h_0, c_0

    def forward(self, z: Tensor, x: Tensor) -> (Tensor, Tensor, Tensor):
        """ Reconstruct a molecule from a latent vector :math:`z` using a conditioned LSTM

        :param z: :math:`(N, Z)`, latent space from variational layer
        :param x: :math:`(N, C)`, true tokens, required for teacher forcing
        :return: sequence_probs: :math:`(N, S, C)`, molecule_loss: :math:`(N)`, loss: :math:`()`, where S = seq. length
        """
        batch_size = z.shape[0]
        seq_len = x.shape[1]

        # Find the token length of all SMILES strings
        length_of_smiles = get_smiles_length_batch(x)

        # init an empty hidden and cell state for the first token
        hidden_state, cell_state = self.condition_lstm(z)

        # init start tokens
        current_token = init_start_tokens(batch_size=batch_size, device=self.device)

        # For every 'current token', generate the next one
        token_probs, token_losses = [], []
        for t_i in range(seq_len - 1):  # loop over all tokens in the sequence

            next_token = x[:, t_i + 1]

            # Embed the starting token
            embedded_token = self.embedding_layer(current_token)

            # next token prediction
            x_hat, (hidden_state, cell_state) = self.lstm(embedded_token, (hidden_state, cell_state))
            logits = F.relu(self.lin_lstm_to_token(x_hat))

            token_loss = self.loss_func(logits.squeeze(), next_token, length_norm=length_of_smiles)
            token_losses.append(token_loss)

            next_token_probs = F.softmax(logits, dim=-1)
            token_probs.append(next_token_probs.squeeze())
            current_token = next_token_probs.argmax(-1)

        # Stack the list of tensors into a single tensor
        token_probs_N_S_C = torch.stack(token_probs, 1)
        token_losses = torch.stack(token_losses, 1)

        # Sum up the token losses to get molecule-wise loss and average out over them to get the overall loss
        molecule_loss = torch.sum(token_losses, 1)
        loss = torch.mean(molecule_loss)

        return token_probs_N_S_C, molecule_loss, loss


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


def init_start_tokens(batch_size: int, device: str = 'cpu') -> Tensor:
    """ Create start one-hot encoded tokens in the shape of (batch size x 1)

    :param start_idx: index of the start token as defined in constants.VOCAB
    :param batch_size: number of molecules in the batch
    :param device: device (default='cpu')
    :return: start token batch tensor
    """
    x = torch.zeros((batch_size, 1), device=device).long()
    x[:, 0] = VOCAB['start_idx']

    return x


def init_lstm_hidden(num_layers: int, batch_size: int, hidden_size: int, device: str) -> (Tensor, Tensor):
    """ Initialize hidden and cell states with zeros

    :return: (Hidden state, Cell state) with shape :math:`(L, N, H)`, where L=num_layers, N=batch_size, H=hidden_size.
    """

    h_0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
    c_0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)

    return h_0, c_0