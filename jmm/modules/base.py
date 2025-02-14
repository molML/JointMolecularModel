
import functools
import torch


class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    def load_weights(self, path: str):
        """ Load a state dict directly from a path"""
        self.load_state_dict(torch.load(path))

    def save_weights(self, path: str):
        """ save the state dict directly to a path"""
        torch.save(self.state_dict(), path)
