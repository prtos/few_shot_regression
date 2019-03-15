import torch
import numpy as np
from torch.nn import Linear, Parameter, ReLU, Sequential, Hardtanh, Module, Sigmoid, Bilinear
from torch.nn.functional import linear
from metalearn.models.utils import *


class GatedConditioning(torch.nn.Module):
    def __init__(self, input_size, condition_size):
        super(GatedConditioning, self).__init__()
        self._output_dim = input_size
        if input_size != condition_size:
            self.downsampling = Linear(condition_size, input_size)
        else:
            self.downsampling = None

    def forward(self, phis, z):
        w = Sigmoid()(z if self.downsampling is None else self.downsampling(z))
        return w * phis

    @property
    def output_dim(self):
        return self._output_dim


class SumConditioning(torch.nn.Module):
    def __init__(self, input_size, condition_size):
        super(SumConditioning, self).__init__()
        self._output_dim = input_size
        if input_size != condition_size:
            self.downsampling = Linear(condition_size, input_size)
        else:
            self.downsampling = None

    def forward(self, phis, z):
        w = (z if self.downsampling is None else self.downsampling(z))
        return w + phis

    @property
    def output_dim(self):
        return self._output_dim


class Film(torch.nn.Module):
    def __init__(self, input_size, condition_size):
        super(Film, self).__init__()
        self.gamma_net = Linear(condition_size, input_size)
        self.beta_net = Linear(condition_size, input_size)
        self.prior_beta_reg_L2 = Parameter(torch.Tensor(input_size))
        self.prior_gamma_reg_L2 = Parameter(torch.Tensor(input_size))
        stdv = 1. / np.sqrt(input_size)
        self.prior_beta_reg_L2.data.uniform_(-stdv, stdv)
        self.prior_gamma_reg_L2.data.uniform_(-stdv, stdv)
        self._output_dim = input_size

    def forward(self, phis, z):
        gamma = self.gamma_net(z)
        beta = self.beta_net(z)
        res = (1 + self.prior_gamma_reg_L2 * gamma) * phis + (self.prior_beta_reg_L2 * beta)
        return res

    @property
    def output_dim(self):
        return self._output_dim


class BilinearNet(torch.nn.Module):
    def __init__(self, input_size, condition_size):
        super(BilinearNet, self).__init__()
        self.net = Bilinear(input_size, condition_size, input_size)
        self._output_dim = input_size

    def forward(self, phis, z):
        return self.net(phis, z)

    @property
    def output_dim(self):
        return self._output_dim


class FusionNet(torch.nn.Module):
    def __init__(self, input_size, condition_size, nb_layers=1, residual=True):
        super(FusionNet, self).__init__()
        hidden_sizes = [int(input_size / 2)] * nb_layers
        in_dim = input_size + condition_size
        fusion_layers = []
        for out_size in hidden_sizes:
            fusion_layers.append(Linear(in_dim, out_size))
            fusion_layers.append(ReLU())
            in_dim = out_size
        fusion_layers.append(Linear(out_size, input_size))
        self.residual = residual
        self.net = Sequential(*fusion_layers)
        self._output_dim = input_size

    def forward(self, phis, z):
        if len(phis.size()) != 2:
            raise ValueError(f'Excepting phis to have two dimensions but got shape {phis.size()}')

        new_inputs = torch.cat((phis, z), dim=1)
        residuals = self.net(new_inputs)
        res = ReLU()(phis + residuals) if self.residual else residuals
        return res

    @property
    def output_dim(self):
        return self._output_dim


class ConditionerFactory:
    def __call__(self, mode, feature_extractor_size, task_descr_encoder_size=None,
                 data_encoder_size=None, **conditioner_params):
        condition_size = 0
        if task_descr_encoder_size is not None:
            condition_size += task_descr_encoder_size
        if data_encoder_size is not None:
            condition_size += data_encoder_size

        if mode.lower() in ['ew', 'film', 'elementwise']:
            conditioner = Film(feature_extractor_size, condition_size)
        elif mode.lower() in ['g', 'gating', 'gate', 'gated']:
            conditioner = GatedConditioning(feature_extractor_size, condition_size)
        elif mode.lower() in ['s', 'sum', 'add', 'a']:
            conditioner = SumConditioning(feature_extractor_size, condition_size)
        elif mode.lower() in ['b', 'bi', 'bilinear']:
            conditioner = BilinearNet(feature_extractor_size, condition_size)
        elif mode.lower() in ['lf', 'fusion', 'cat', 'concat', 'latefusion']:
            conditioner = FusionNet(feature_extractor_size, condition_size, **conditioner_params)
        else:
            raise Exception('mode {} is not allowed. The value should be: ew, tf, lf '.format(mode))

        return conditioner
