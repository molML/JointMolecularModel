import torch
from torch import Tensor


def to_binary(x: torch.Tensor, threshold: float = 0.5):
    return torch.where(x > threshold, torch.tensor(1), torch.tensor(0))


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

    def __init__(self, y, y_hat):
        self.n = len(y)
        self.pp = sum(y_hat).item()
        self.TP, self.TN, self.FP, self.FN = confusion_matrix(y, y_hat)

        self.TPR = self.TP / sum(y).item()  # true positive rate, hit rate, recall, sensitivity
        self.FNR = 1 - self.TPR  # false negative rate, miss rate
        self.TNR = self.TN / sum(y == 0).item()  # true negative rate, specificity, selectivity
        self.FPR = 1 - self.TNR  # false positive rate, fall-out

    def accuracy(self):
        self.ACC = (self.TP + self.TN) / (len(y))
        return self.ACC

    def balanced_accuracy(self):
        self.BA = (self.TPR + self.TNR) / 2  # balanced accuracy
        return self.BA

    def precision(self):
        self.PPV = self.TP / self.pp  # precision
        return self.PPV

    def f1(self):
        self.precision()
        self.F1 = (2*self.PPV * self.TPR) / (self.PPV + self.TPR)
        return self.F1

    def __repr__(self):
        return f"TP: {self.TP}, TN: {self.TN}, FP: {self.FP}, FN: {self.FN}\n" \
               f"TPR: {round(self.TPR, 4)}, FNR: {round(self.FNR, 4)}, TNR: {round(self.TNR, 4)}, " \
               f"FPR: {round(self.FPR, 4)}"
