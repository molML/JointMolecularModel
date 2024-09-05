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


class AutoregressiveRNN(nn.Module):
    """ An autoregressive RNN that takes integer-encoded SMILES strings and performs next token prediction.
    Negative Log Likelihood is calculated per molecule and per batch and is normalized per molecule length so that
    small molecules do not get an unfair advantage in loss over longer ones.

    :param rnn_hidden_size: size of the RNN hidden layers (default=256)
    :param vocabulary_size: size of the vocab (default=36)
    :param rnn_num_layers: number of RNN layers (default=2)
    :param rnn_embedding_dim: size of the SMILES embedding layer (default=128)
    :param rnn_dropout: dropout ratio, num_layers should be > 1 if dropout > 0 (default=0.2)
    :param device: device (default='cpu')
    :param ignore_index: index of the padding token (default=35, padding tokens must be ignored in this implementation)
    """

    def __init__(self, rnn_hidden_size: int = 256, vocabulary_size: int = 36, rnn_num_layers: int = 2,
                 token_embedding_dim: int = 128, ignore_index: int = 0, rnn_dropout: float = 0.2, device: str = 'cpu',
                 rnn_type: str = 'gru', **kwargs) -> None:
        super(AutoregressiveRNN, self).__init__()

        assert rnn_type in ['gru', 'lstm'], f"rnn_type should be 'gru' or 'lstm', not 'f{rnn_type}'."

        self.hidden_size = rnn_hidden_size
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = token_embedding_dim
        self.num_layers = rnn_num_layers
        self.device = device
        self.ignore_index = ignore_index
        self.dropout = rnn_dropout
        self.rnn_type = rnn_type

        self.loss_func = nn.NLLLoss(reduction='none', ignore_index=ignore_index)

        self.embedding_layer = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=token_embedding_dim)

        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=token_embedding_dim, hidden_size=rnn_hidden_size, batch_first=True,
                              num_layers=self.num_layers, dropout=rnn_dropout)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=token_embedding_dim, hidden_size=rnn_hidden_size, batch_first=True,
                               num_layers=self.num_layers, dropout=rnn_dropout)

        self.fc = nn.Linear(in_features=rnn_hidden_size, out_features=vocabulary_size)

    def forward(self, x: Tensor, *args) -> (Tensor, Tensor, Tensor, Tensor):
        """ Perform next-token autoregression on a batch of SMILES strings

        :param x: :math:`(N, S)`, integer encoded SMILES strings where S is sequence length, as .long()
        :param args: redundant param that is kept for compatability
        :return:  predicted token probability, molecule embedding, molecule loss, batch loss
        """

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # turn indexed encoding into embeddings
        embedding = self.embedding_layer(x)

        # init an empty hidden and cell state for the first token
        hidden_state = init_rnn_hidden(num_layers=self.num_layers, batch_size=batch_size, hidden_size=self.hidden_size,
                                       device=self.device, rnn_type=self.rnn_type)

        mol_loss = 0  # will become (N)
        all_log_probs = []
        for t_i in range(seq_len - 1):  # loop over all tokens in the sequence

            # Get the current and next token in the sequence
            target_tokens = x[:, t_i + 1]  # (batch_size)
            current_tokens = embedding[:, t_i, :]  # (batch_size, 1, vocab_size)

            # predict the next token in the sequence
            logits, hidden_state = self.rnn(current_tokens.unsqueeze(1), hidden_state)
            logits = self.fc(logits)  # (batch_size, 1, vocab_size)

            log_probs = F.log_softmax(logits, dim=-1)  # (N, 1, C)
            all_log_probs.append(log_probs)

            mol_loss += self.loss_func(log_probs.squeeze(1), target_tokens)

        # Get the mini-batch loss
        loss = torch.mean(mol_loss)  # ()

        # Normalize molecule loss by molecule size. # Find the position of the first occuring padding token, which is
        # the length of the SMILES
        mol_loss = mol_loss / get_smiles_length_batch(x)  # (N)

        # concat all individual token log probs over the sequence dimension to get to one big tensor
        all_log_probs_N_S_C = torch.cat(all_log_probs, 1)  # (N, S-1, C)

        return all_log_probs_N_S_C, mol_loss, loss


class DecoderRNN(nn.Module):

    def __init__(self, rnn_hidden_size: int = 256, vocabulary_size: int = 36, rnn_num_layers: int = 2,
                 token_embedding_dim: int = 128, z_size: int = 128, ignore_index: int = 0, rnn_dropout: float = 0.2,
                 device: str = 'cpu', rnn_type: str = 'gru', rnn_teacher_forcing: bool = False, **kwargs) -> None:
        super(DecoderRNN, self).__init__()

        self.hidden_size = rnn_hidden_size
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = token_embedding_dim
        self.num_layers = rnn_num_layers
        self.device = device
        self.ignore_index = ignore_index
        self.dropout = rnn_dropout
        self.rnn_type = rnn_type
        self.z_size = z_size
        self.teacher_forcing = rnn_teacher_forcing

        self.loss_func = nn.NLLLoss(reduction='none', ignore_index=ignore_index)

        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=token_embedding_dim, hidden_size=rnn_hidden_size, batch_first=True,
                              num_layers=self.num_layers, dropout=rnn_dropout)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=token_embedding_dim, hidden_size=rnn_hidden_size, batch_first=True,
                               num_layers=self.num_layers, dropout=rnn_dropout)

        self.z_transform = nn.Linear(in_features=z_size, out_features=rnn_hidden_size * rnn_num_layers)
        self.lin_rnn_to_token = nn.Linear(in_features=rnn_hidden_size, out_features=vocabulary_size)
        self.embedding_layer = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=token_embedding_dim)

    def condition_rnn(self, z: Tensor) -> (Tensor, Tensor):
        """ Condition the initial hidden state of the rnn with a latent vector z

        :param z: :math:`(N, Z)`, batch of latent molecule representations
        :return: :math:`(L, N, H), (L, N, H)`, hidden state & cell state, where L is num_layers, H is rnn hidden size
        """

        batch_size = z.shape[0]
        # transform z to rnn_hidden_size * rnn_num_layers
        z = F.relu(self.z_transform(z))

        # reshape z into the rnn hidden state so it's distributed over the num_layers. This makes sure that for each
        # item in the batch, it's split into num_layers chunks, with shape (num_layers, batch_size, hidden_size) so
        # that the conditioned information is still matched for each item in the batch
        h_0 = z.reshape(batch_size, self.num_layers, self.hidden_size).transpose(1, 0).contiguous()

        if self.rnn_type == 'gru':
            return h_0
        elif self.rnn_type == 'lstm':
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
            return h_0, c_0

    def forward(self, z: Tensor, x: Tensor) -> (Tensor, Tensor, Tensor):
        """ Reconstruct a molecule from a latent vector :math:`z` using a conditioned rnn

        :param z: :math:`(N, Z)`, latent space from variational layer
        :param x: :math:`(N, C)`, true tokens, required for teacher forcing
        :return: sequence_probs: :math:`(N, S, C)`, molecule_loss: :math:`(N)`, loss: :math:`()`, where S = seq. length
        """
        batch_size = z.shape[0]
        seq_len = x.shape[1]

        # init an empty hidden and cell state for the first token
        hidden_state = self.condition_rnn(z)

        # init start tokens
        current_token = init_start_tokens(batch_size=batch_size, device=self.device)

        # For every 'current token', generate the next one
        mol_loss = 0  # will become (N)
        all_log_probs = []
        for t_i in range(seq_len - 1):  # loop over all tokens in the sequence

            target_tokens = x[:, t_i + 1]

            # Embed the starting token
            embedded_token = self.embedding_layer(current_token)

            # next token prediction
            logits, hidden_state = self.rnn(embedded_token, hidden_state)
            logits = self.lin_rnn_to_token(logits)

            log_probs = F.log_softmax(logits, dim=-1)  # (N, 1, C)
            all_log_probs.append(log_probs)

            mol_loss += self.loss_func(log_probs.squeeze(1), target_tokens)

            if self.teacher_forcing:
                current_token = target_tokens.unsqueeze(1)
            else:
                current_token = log_probs.argmax(-1)

        # Get the mini-batch loss
        loss = torch.mean(mol_loss)  # ()

        # Normalize molecule loss by molecule size.
        mol_loss = mol_loss / get_smiles_length_batch(x)  # (N)

        # concat all individual token log probs over the sequence dimension to get to one big tensor
        all_log_probs_N_S_C = torch.cat(all_log_probs, 1)  # (N, S-1, C)

        return all_log_probs_N_S_C, mol_loss, loss

    def generate_from_z(self, z, seq_len: int = 101):
        """ Reconstruct a molecule from a latent vector :math:`z` using a conditioned rnn

       :param z: :math:`(N, Z)`, latent space from variational layer
       :param seq_len: number of character tokens you want to generate
       :return: sequence_probs: :math:`(N, S, C)`
       """
        batch_size = z.shape[0]

        # init an empty hidden and cell state for the first token
        hidden_state = self.condition_rnn(z)

        # init start tokens
        current_token = init_start_tokens(batch_size=batch_size, device=self.device)

        # For every 'current token', generate the next one
        all_log_probs = []
        for t_i in range(seq_len - 1):  # loop over all tokens in the sequence

            # Embed the starting token
            embedded_token = self.embedding_layer(current_token)

            # next token prediction
            logits, hidden_state = self.rnn(embedded_token, hidden_state)
            logits = self.lin_rnn_to_token(logits)

            log_probs = F.log_softmax(logits, dim=-1)  # (N, 1, C)
            all_log_probs.append(log_probs)

            current_token = log_probs.argmax(-1)

        # concat all individual token log probs over the sequence dimension to get to one big tensor
        all_log_probs_N_S_C = torch.cat(all_log_probs, 1)  # (N, S-1, C)

        return all_log_probs_N_S_C


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


def init_rnn_hidden(num_layers: int, batch_size: int, hidden_size: int, device: str, rnn_type: str = 'gru'):  # -> Tensor | tuple[Tensor, Tensor]:
    """ Initialize hidden and cell states with zeros. rnn_type can be either 'gru' or 'lstm

    :return: (Hidden state, Cell state) with shape :math:`(L, N, H)`, where L=num_layers, N=batch_size, H=hidden_size.
    """

    assert rnn_type in ['gru', 'lstm'], f"rnn_type should be 'gru' or 'lstm', not 'f{rnn_type}'."

    h_0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)

    if rnn_type == 'gru':
        return h_0

    elif rnn_type == 'lstm':
        c_0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        return h_0, c_0
