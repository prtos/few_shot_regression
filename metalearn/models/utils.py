import torch, subprocess
import numpy as np
import operator as op
from functools import reduce
from collections import OrderedDict
from torch.nn import MSELoss
from torch.nn.functional import mse_loss


def to_numpy_vec(x):
    return x.data.cpu().numpy().flatten()


def to_unit(t):
    if isinstance(t, torch.Tensor):
        x = t.data.cpu().numpy()
    else:
        x = t
    return x
    

def prod(iterable):
    return reduce(op.mul, iterable, 1)


def reparameterize(mu, var, is_training):
    if is_training and var is not None:
        std = torch.std(var)
        eps = torch.randn_like(std)
        return mu + (std * eps)
    else:
        return mu


def set_params(module, new_params, prefix=''):
    """
    This allows to set the params of a module without messing with the variables and thus incapacitate the backprop
    :param module: the module of interest
    :param new_params: the params new params of the module.
            see torch.nn.Module.named_parameters() for the format of this arg
    :param prefix: default '', otherwise the name of the module
    :return:
    """
    module._parameters = OrderedDict((name, new_params[prefix + ('.' if prefix else '') + name])
                                     for name, _ in module._parameters.items())
    for mname, submodule in module.named_children():
        submodule_prefix = prefix + ('.' if prefix else '') + mname
        set_params(submodule, new_params, submodule_prefix)





def reset_BN_stats(module):
    """
    Erase the running statistics of the batch norm layers if the module uses them
    :param module: the module of interest
    :return:
    """
    for module in module.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))\
                and module.track_running_stats:
            # pass
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)


def KL_div_diag_multivar_normals(mu1, var1, mu2=None, var2=None):
    """
    Compute the KL-divergence between 2 multivariate gaussian distributions with diagonal covariance matrices.
    For computation details, see: stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    :param mu1: The expected values of the first distribution
    :param logvar1: The log of the diagonal of the covariance matrix of the first distribution
    :param mu2: The expected values of the second distribution
    :param logvar2: The log of the diagonal of the covariance matrix of the second distribution
    :return: The value of KL(Normal(mu1, logvar1), Normal(mu2, logvar2))
    """
    assert mu1 is not None, "mu1 should not be None"
    if mu2 is None:
        mu2 = torch.zeros_like(mu1)
    if var2 is None:
        var2 = torch.ones_like(mu1)
    if var1 is None:
        var1 = torch.ones_like(mu1)
    logvar1 = torch.log(var1)
    logvar2 = torch.log(var2)
    r = ((mu1 - mu2).pow(2) + logvar1.exp())/(2 * logvar2.exp()) + 0.5*(logvar2 - logvar1) - 0.5
    kl = torch.sum(r)
    return kl


def normal_entropy(mu, var):
    logvar = torch.log(var)
    return logvar.sum() + 0.5*logvar.size(0)*np.log(2*np.pi*np.e)


def sigm_heating(t, max_x=1.0, max_t=2000.0):
    res = max_x/(1 + (1/0.01 - 1)*np.exp(-t/max_t))
    return res


def annealed_softmax(x, t, cooling_factor=0.001):
    y = x - x.max()
    beta = min(t*cooling_factor, 1)
    res = beta*torch.exp(y) + (1 - beta)*(1/x.size(-1))
    res = res / res.sum(dim=-1)
    return res


class MaskedMSE(MSELoss):
    def __init__(self, size_average=True, reduce=True):
        super(MaskedMSE, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target_and_mask):
        target, mask = target_and_mask
        non_zeros = torch.nonzero(mask)
        y_true = target[non_zeros[:, 0], non_zeros[:, 1]]
        y_pred = input[non_zeros[:, 0], non_zeros[:, 1]]
        return mse_loss(y_pred, y_true, size_average=self.size_average, reduce=self.reduce)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    max = 0
    f = lambda x: max/(1 + (1/0.01 - 1)*np.exp(-x/2000.0))
    x = np.arange(20000)
    y = np.apply_along_axis(f, axis=0, arr=x)
    print(np.min(y))
    plt.plot(x, y)
    plt.show()

