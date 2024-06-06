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


class AutoregressiveLSTM(nn.Module):
    """ An autoregressive LSTM that takes integer-encoded SMILES strings and performs next token prediction.
    Negative Log Likelihood is calculated per molecule and per batch and is normalized per molecule length so that
    small molecules do not get an unfair advantage in loss over longer ones.

    :param hidden_size: size of the LSTM hidden layers (default=256)
    :param vocab_size: size of the vocab (default=36)
    :param num_layers: number of LSTM layers (default=1)
    :param embedding_dim: size of the SMILES embedding layer (default=128)
    :param ignore_index: index of the padding token (default=35, padding tokens must be ignored in this implementation)
    """

    def __init__(self, hidden_size: int = 256, vocab_size: int = 36, num_layers: int = 1, embedding_dim: int = 128,
                 ignore_index: int = 35) -> None:
        super(AutoregressiveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.loss_func = nn.NLLLoss(reduction='none', ignore_index=ignore_index)

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True,
                            num_layers=self.num_layers)
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def init_hidden(self, batch_size):
        # Initialize hidden and cell states with zeros
        return (torch.zeros(1, batch_size, self.hidden_size), torch.zeros(1, batch_size, self.hidden_size))

    def forward(self, x: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        """ Perform next-token autoregression on a batch of SMILES strings

        :param x: integer encoded SMILES strings (batch_size x sequence_length), as .long()
        :return:  predicted token probability, molecule embedding, molecule loss, batch loss
        """

        # find the position of the first occuring padding token, which is the length of the SMILES
        length_of_smiles = get_smiles_length_batch(x)
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # turn indexed encoding into embeddings
        embedding = self.embedding_layer(x)

        # init an empty hidden and cell state for the first token
        hidden_state, cell_state = self.init_hidden(batch_size=batch_size)

        loss, probs = [], []
        for t_i in range(seq_len - 1):  # loop over all tokens in the sequence

            # Get the current and next token in the sequence
            x_i = embedding[:, t_i, :].unsqueeze(1)  # (batch_size, 1, vocab_size)
            next_token = x[:, t_i + 1]  # (batch_size, vocab_size)

            # predict the next token in the sequence
            x_hat, (hidden_state, cell_state) = self.lstm(x_i, (hidden_state, cell_state))
            logits = F.relu(self.fc(x_hat))  # (batch_size, 1, vocab_size)

            # Compute loss
            x_probs = F.log_softmax(logits, dim=-1).squeeze()  # (batch_size, vocab_size)
            loss_i = self.loss_func(x_probs, next_token)  # (batch_size)

            probs.append(x_probs)
            loss.append(loss_i)

        # stack the token-wise losses and the predicted token probabilities
        loss = torch.stack(loss, 1)
        probs = torch.stack(probs, 1)

        # Sum up the losses (padding should be ignored so this doesn't affect the sum) and normalize by SMILES length
        # This is a mean Loss that is not biased by SMILES length. If you don't do this, longer SMILES get penalized.
        sample_loss = torch.sum(loss, 1) / length_of_smiles

        return probs, embedding, sample_loss, torch.mean(sample_loss)
