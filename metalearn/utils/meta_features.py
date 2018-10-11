import torch
import numpy as np
import torch.functional as F
from few_shot_regression.models.krr import *


def entropy(arr, arr2=None):
    if arr2 is None:
        c = torch.histc(arr, bins=10)
    else:
        c = np.histogram2d(arr, arr2)[0]
    b = F.softmax(c) * F.log_softmax(c)
    H = -1.0 * b.sum()
    return H


def mutual_information(x, y):
    return entropy(x) + entropy(y) - entropy(x, y)


def apply_along_axis(func, tensor, dim):
    return torch.stack([func(m) for m in torch.unbind(tensor, dim=dim)], dim=0)


def get_meta_features(inputs, targets):
    # Cross-Disciplinary Perspectives on Meta-Learning for Algorithm Selection KATE A. SMITH-MILES
    total_variation = torch.std(inputs, axis=0) / (torch.mean(inputs, axis=0) + 1e-4)
    class_entropy = entropy(targets)
    inputs_entropy = torch.mean(apply_along_axis(entropy, inputs, dim=0))
    mi_class_attr = torch.mean(apply_along_axis(lambda x: entropy(targets, x), inputs, dim=0))
