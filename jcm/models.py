
import torch
from torch import nn
from jcm.modules.lstm import AutoregressiveLSTM
from constants import VOCAB
import torch.nn.functional as F
from dataprep.descriptors import encoding_to_smiles


class DeNovoLSTM(AutoregressiveLSTM):
    # SMILES -> LSTM -> SMILES

    def __init__(self):
        super(DeNovoLSTM, self).__init__()
        self.device = 'cpu'

    @torch.no_grad()
    def generate(self, design_length: int = 102, batch_size: int = 256, temperature: int = 1, sample: bool = True):

        # init start tokens and add them to the list of generated tokens
        current_token = self.init_start_tokens(batch_size=batch_size)
        tokens = [current_token.squeeze()]

        # init an empty hidden and cell state for the first token
        hidden_state, cell_state = self.init_hidden(batch_size=batch_size)

        # For every 'current token', generate the next one
        for t_i in range(design_length - 1):  # loop over all tokens in the sequence

            # Get the SMILES embeddings
            x_i = self.embedding_layer(current_token)

            # next token prediction
            x_hat, (hidden_state, cell_state) = self.lstm(x_i, (hidden_state, cell_state))
            logits = F.relu(self.fc(x_hat))

            # perform temperature scaling
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            # Get the next token
            if sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = torch.topk(probs, k=1, dim=-1)

            # update the 'current token' and the list of generated tokens
            tokens.append(next_token.squeeze())
            current_token = next_token

        tokens = torch.stack(tokens, 1)
        smiles = encoding_to_smiles(tokens)

        return smiles

    @torch.no_grad()
    def predict(self):
        pass

    @staticmethod
    def callback():
        pass

    def init_start_tokens(self, batch_size: int):
        x = torch.zeros((batch_size, 1), device=self.device).long()
        x[:, 0] = VOCAB['start_idx']

        return x



class EcfpMLP(nn.Module):
    # ECFP -> MLP -> property
    def __init__(self):
        super(EcfpMLP, self).__init__()

    @torch.no_grad()
    def predict(self):
        pass

    @staticmethod
    def callback():
        pass


class SmilesMLP(nn.Module):
    # smiles -> CNN -> variational -> MLP -> property
    def __init__(self):
        super(SmilesMLP, self).__init__()

    @torch.no_grad()
    def predict(self):
        pass

    @staticmethod
    def callback():
        pass


class VAE(nn.Module):
    # SMILES -> CNN -> variational -> LSTM -> SMILES
    @torch.no_grad()
    def generate(self):
        pass

    @torch.no_grad()
    def predict(self):
        pass

    @staticmethod
    def callback():
        pass


class JointChemicalModel(nn.Module):
    # SMILES -> CNN -> variational -> LSTM -> SMILES
    #                            |
    #                           MLP -> property
    @torch.no_grad()
    def generate(self):
        pass

    @torch.no_grad()
    def predict(self):
        pass

    @staticmethod
    def callback():
        pass
