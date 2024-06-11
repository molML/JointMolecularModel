
import functools
import torch


class BaseModule(torch.nn.Module):
    """
    Enables the use of an @inference decorator that combines model.eval and torch.no_grad()
    """
    def __init__(self):
        super(BaseModule, self).__init__()

    def inference_(self, func):
        """ an inference decorator that combines model.eval and torch.no_grad() """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Save the original mode of the model
            original_mode = self.training
            try:
                # Set the model to evaluation mode
                self.eval()
                # Execute the function within torch.no_grad() context
                with torch.no_grad():
                    result = func(*args, **kwargs)
            finally:
                # Restore the original mode of the model
                self.train(original_mode)
            return result
        return wrapper

    @property
    def inference(self):
        """ an inference decorator that combines model.eval and torch.no_grad() """
        return self.inference_

    def load_weights(self, path: str):
        """ Load a state dict directly from a path"""
        self.load_state_dict(torch.load(path))

    def save_weights(self, path: str):
        """ save the state dict directly to a path"""
        torch.save(self.state_dict(), path)
