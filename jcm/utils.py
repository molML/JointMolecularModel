
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader, default_collate
from cheminformatics.encoding import encoding_to_smiles
from constants import VOCAB


def to_binary(x: torch.Tensor, threshold: float = 0.5):
    return torch.where(x > threshold, torch.tensor(1), torch.tensor(0))


def scale_BCE(loss: Tensor, y: Tensor, factor: float = 1) -> Tensor:
    """ Scales BCE loss for the 1 class, i.e., a scaling factor of 2 would weight the '1' class twice as heavy as the
    '0' class. Requires BCE loss with reduction='none'

    :param loss: BCELoss with reduction='none'
    :param y: labels, in the same shape as the loss (which happens with reduction='none')
    :param factor: scaling factor (default=1)
    :return: scaled loss
    """
    scaling_tensor = torch.ones(loss.shape)
    scaling_tensor[y == 1] = factor

    return loss * scaling_tensor


def BCE_per_sample(y_hat: Tensor, y: Tensor, class_scaling_factor: float = None) -> (Tensor, Tensor):
    """ Computes the BCE loss and also returns the summed BCE per individual samples

    :param y_hat: predictions [batch_size, binary bits]
    :param y: labels [batch_size, binary bits]
    :param class_scaling_factor: Scales BCE loss for the '1' class, i.e. a factor of 2 would double the respective loss
        for the '1' class (default=None)
    :return: overall batch BCE loss, BCE per sample
    """

    loss_fn = nn.BCELoss(reduction='none')
    loss = loss_fn(y_hat, y.float())
    sample_loss = torch.mean(loss, 1)

    if class_scaling_factor is not None:
        loss = scale_BCE(loss, y, factor=class_scaling_factor)

    return torch.mean(loss), sample_loss


def logits_to_pred(logprobs_N_K_C: Tensor, return_binary: bool = False, return_uncertainty: bool = True) -> (Tensor, Tensor):
    """ Get the probabilities/class vector and sample uncertainty from the logits """

    mean_probs_N_C = torch.mean(torch.exp(logprobs_N_K_C), dim=1)
    uncertainty = mean_sample_entropy(logprobs_N_K_C)

    if not return_binary:
        y_hat = mean_probs_N_C
    else:
        y_hat = torch.argmax(mean_probs_N_C, dim=1)

    if return_uncertainty:
        return y_hat, uncertainty
    else:
        return y_hat


def logits_to_smiles(logits: Tensor) -> list[str]:
    """ Convert logits back to SMILES by softmaxing and taking most probable token

    :param logits: tensor of shape (batch_size, sequence_length, vocab)
    :return: list of SMILES strings decoded according to the vocab
    """
    token_probs = F.softmax(logits, -1)
    token_idx_batch = token_probs.argmax(-1)

    designs = [''.join([VOCAB['indices_token'][int(token_idx)] for token_idx in row]) for row in token_idx_batch]

    return designs


def logit_mean(logprobs_N_K_C: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """ Logit mean with the logsumexp trick - Kirch et al., 2019, NeurIPS """

    return torch.logsumexp(logprobs_N_K_C, dim=dim, keepdim=keepdim) - math.log(logprobs_N_K_C.shape[dim])


def entropy(logprobs_N_K_C: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """Calculates the Shannon Entropy """

    return -torch.sum((torch.exp(logprobs_N_K_C) * logprobs_N_K_C).double(), dim=dim, keepdim=keepdim)


def mean_sample_entropy(logprobs_N_K_C: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    """Calculates the mean entropy for each sample given multiple ensemble predictions - Kirch et al., 2019, NeurIPS"""

    sample_entropies_N_K = entropy(logprobs_N_K_C, dim=dim, keepdim=keepdim)
    entropy_mean_N = torch.mean(sample_entropies_N_K, dim=1)

    return entropy_mean_N


def mutual_information(logprobs_N_K_C: Tensor) -> Tensor:
    """ Calculates the Mutual Information - Kirch et al., 2019, NeurIPS """

    # this term represents the entropy of the model prediction (high when uncertain)
    entropy_mean_N = mean_sample_entropy(logprobs_N_K_C)

    # This term is the expectation of the entropy of the model prediction for each draw of model parameters
    mean_entropy_N = entropy(logit_mean(logprobs_N_K_C, dim=1), dim=-1)

    I = mean_entropy_N - entropy_mean_N

    return I


def confusion_matrix(y: Tensor, y_hat: Tensor) -> (float, float, float, float):
    """ Compute a confusion matrix from binary classification predictions

    :param y: tensor of true labels
    :param y_hat: tensor of predicted
    :return: TP, TN, FP, FN
    """
    TP = sum(y_hat[y == 1])
    TN = sum(y == 0) - sum(y_hat[y == 0])
    FP = sum(y_hat[y == 0])
    FN = len(y_hat[y == 1]) - sum(y_hat[y == 1])

    return TP.item(), TN.item(), FP.item(), FN.item()


class ClassificationMetrics:
    TP, TN, FP, FN = None, None, None, None
    ACC, BA, PPV, F1 = None, None, None, None
    eps = 1e-6  # prevents divide by zeros

    def __init__(self, y, y_hat):
        self.n = len(y)
        self.pp = sum(y_hat).item()
        self.pp_true = sum(y).item()
        self.TP, self.TN, self.FP, self.FN = confusion_matrix(y, y_hat)

        self.TPR = self.TP / (sum(y).item() + self.eps)  # true positive rate, hit rate, recall, sensitivity
        self.FNR = 1 - self.TPR  # false negative rate, miss rate
        self.TNR = self.TN / (sum(y == 0).item() + self.eps)  # true negative rate, specificity, selectivity
        self.FPR = 1 - self.TNR  # false positive rate, fall-out

        self.recons = sum(y_hat == y).item() / self.n

        self.accuracy()
        self.balanced_accuracy()
        self.precision()
        self.f1()
        self.reconstruction()

    def accuracy(self):
        self.ACC = (self.TP + self.TN) / self.n
        return self.ACC

    def reconstruction(self):
        self.recons_100 = (self.recons == 1)*1
        self.recons_99 = (self.recons >= 0.99)*1
        self.recons_95 = (self.recons >= 0.95)*1

    def balanced_accuracy(self):
        self.BA = (self.TPR + self.TNR) / 2  # balanced accuracy
        return self.BA

    def precision(self):
        self.PPV = self.TP / (self.pp + self.eps)  # precision
        return self.PPV

    def f1(self):
        self.precision()
        self.F1 = (2*self.PPV * self.TPR) / (self.PPV + self.TPR + self.eps)
        return self.F1

    def all(self):
        return self.__dict__

    def __repr__(self):

        balance = f"Balance y: {round(self.pp_true*100 / self.n, 4)}, y_hat: {round(self.pp * 100 / self.n, 4)}\n"
        confusion = f"TP: {self.TP}, TN: {self.TN}, FP: {self.FP}, FN: {self.FN}\n"
        rates = f"TPR: {round(self.TPR, 4)}, FNR: {round(self.FNR, 4)}, TNR: {round(self.TNR, 4)}, " \
                f"FPR: {round(self.FPR, 4)}\n"
        metrics = f"ACC: {round(self.ACC, 4)}, BA: {round(self.BA, 4)}, Precision: {round(self.PPV, 4)}, F1: " \
                  f"{round(self.F1, 4)}"

        return balance + confusion + rates + metrics


def get_val_loader(config, dataset, batch_size: int = 256, sample: bool = False):
    if sample:
        num_samples = config.val_molecules_to_sample
        val_loader = DataLoader(dataset,
                                sampler=RandomSampler(dataset, replacement=True, num_samples=num_samples),
                                shuffle=False, pin_memory=True, batch_size=batch_size,
                                collate_fn=single_batchitem_fix)
    else:
        val_loader = DataLoader(dataset, sampler=None, shuffle=False, pin_memory=True, batch_size=batch_size,
                                collate_fn=single_batchitem_fix)

    return val_loader


def single_batchitem_fix(batch):

    has_xy = True if len(batch[0]) > 1 else False
    is_single_batchitem = True if len(batch) == 1 else False

    if has_xy:
        x, y = default_collate(batch)
        x = x.squeeze(0 if is_single_batchitem else 1).long()
        y = y.squeeze(0 if is_single_batchitem else 1)
        batch = x, y
    else:
        batch = default_collate(batch).squeeze(0 if is_single_batchitem else 1).long()

    return batch


def rnn_output_to_smiles(x):
    return [encoding_to_smiles(enc.tolist()) for enc in x.argmax(dim=1)]


def load_model(model, config, state_dict_path):
    f = model(**config.hyperparameters)
    f.load_state_dict(torch.load(state_dict_path))
    f = f.to(f.device)

    return f


def get_smiles_length_batch(x: torch.Tensor) -> torch.Tensor:
    """ Find out how many tokens a SMILES strings contains by finding the position of the first padding token

    :param x: SMILES tokens either one-hot or integer encoding
    :return: tensor of SMILES lengths (batch_size)
    """
    if x.dim() == 3:
        return x.argmax(2).argmax(1)
    elif x.dim() == 2:
        return x.argmax(1)
    else:
        raise ValueError(f"Cannot find the length of SMILES for {type(x)} shape {x.shape}")


def batch_management(batch, device: str = 'cpu') -> (Tensor, Tensor):
    """ Returns a tuple of (x, y) regardless if the batch itself is a tuple or just a tensor. Also moves stuff to the
    correct device

    :param batch: either a single :math:`x` Tensor or a :math:`(x, y)` tuple of two Tensors
    :param device: torch device (default='cpu')
    :return: :math:`(x, y)` tensors, where y = None if the input was singular
    """

    if type(batch) is tuple and len(batch) == 2:
        return batch[0].to(device), batch[1].to(device)
    else:
        return batch.to(device), None

