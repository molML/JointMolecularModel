
import torch
from torch import nn
from jcm.modules.lstm import AutoregressiveLSTM


class DeNovoLSTM(AutoregressiveLSTM):
    # SMILES -> LSTM -> SMILES

    def __init__(self):
        super(DeNovoLSTM, self).__init__()

    @torch.no_grad()
    def generate(self, temperature: int = 1):

        # logits, _ = self(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        # logits = logits[:, -1, :] / temperature

        # probs = F.softmax(logits, dim=-1)
        # # either sample from the distribution or take the most likely element
        # if do_sample:
        #     idx_next = torch.multinomial(probs, num_samples=1)

        pass

    @torch.no_grad()
    def predict(self):
        pass

    @staticmethod
    def callback():
        pass


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
