import torch
import operator as op
from torch.nn import MSELoss, ModuleList, Linear, Sequential, ReLU
from torch.optim import Adam
from functools import reduce
from pytoune.framework import Model
from .base import MetaLearnerRegression
from .utils import set_params, OrderedDict, reset_BN_stats
from few_shot_regression.utils.feature_extraction import ClonableModule
from .krr import *


def prod(iterable):
    return reduce(op.mul, iterable, 1)


def has_batchnorm(network):
    for module in network.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            return True
    return False


class HyperNetworkFull(torch.nn.Module):
    def __init__(self, base_network: ClonableModule, input_size):
        super(HyperNetworkFull, self).__init__()
        self.base_network = base_network
        self.net = ModuleList()
        self.names = []
        for name, params in self.base_network.named_parameters():
            n = prod(params.size())
            self.net.append(Linear(input_size, n))
            self.names.append(name)

    def __forward(self, z):
        new_network = self.base_network.clone()
        new_weigths = OrderedDict(
            (name, self.net[i](z).reshape(params.size()))
            for i, (name, params) in enumerate(self.base_network.named_parameters()))
        set_params(new_network, new_weigths)
        return new_network

    def forward(self, zs):
        return [self.__forward(z) for z in zs]


class ConditionnalBatchNorm(torch.nn.Module):
    def __init__(self, base_network: ClonableModule, input_size, emb_size):
        super(ConditionnalBatchNorm, self).__init__()
        self.base_network = base_network
        self.input_size = input_size
        self.emb_size = emb_size
        nb_batch_norm = 0
        print(base_network)
        for module in self.base_network.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                nb_batch_norm += module.num_features
        self.out_size = nb_batch_norm

        if self.out_size != 0:
            self.beta_net = Sequential(
                Linear(self.input_size, self.emb_size),
                ReLU(),
                Linear(self.emb_size, self.out_size),
                )

            self.gamma_net = Sequential(
                Linear(self.input_size, self.emb_size),
                ReLU(),
                Linear(self.emb_size, self.out_size),
                )
            # normalize for zeros mean and one variance

    def __forward(self, z):
        betas = self.beta_net(z)
        gammas = self.gamma_net(z)
        new_network = self.base_network.clone()
        new_weights = OrderedDict()
        pos = 0
        for name, params in self.base_network.named_parameters():
            if ('BatchNorm' in name) and name.endswith('bias'):
                num_features = prod(params.size())
                new_weights[name] = params + betas[pos: pos+num_features].reshape(params.size())
            elif ('BatchNorm' in name) and name.endswith('weight'):
                num_features = prod(params.size())
                new_weights[name] = params + gammas[pos: pos+num_features].reshape(params.size())
            else:
                new_weights[name] = params + 0.0

        set_params(new_network, new_weights)
        return new_network

    def forward(self, zs):
        return [self.__forward(z) for z in zs]