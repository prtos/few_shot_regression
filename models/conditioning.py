import torch
import numpy as np
from torch.nn import Linear, Parameter, ReLU, Sequential, Hardtanh, Tanh
from torch.nn.functional import linear
from .utils import *


def compute_data_statistics(phis, y):
    res = torch.mm(y.t(), phis)
    # var = [phis.mean(dim=0), phis.std(dim=0), phis.max(dim=0)[0], phis.min(dim=0)[0], corr[0],
    #                y.mean(dim=0), y.std(dim=0), y.max(dim=0)[0], y.min(dim=0)[0]
    #        ]
    # res = torch.cat(var, 0)
    return res


class Film(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Film, self).__init__()
        self.gamma_net = Linear(input_size, output_size)
        self.beta_net = Linear(input_size, output_size)
        self.prior_beta_reg_L2 = Parameter(torch.Tensor(output_size))
        self.prior_gamma_reg_L2 = Parameter(torch.Tensor(output_size))
        stdv = 1. / np.sqrt(output_size)
        self.prior_beta_reg_L2.data.uniform_(-stdv, stdv)
        self.prior_gamma_reg_L2.data.uniform_(-stdv, stdv)

    def forward(self, phis, z):
        gamma = self.gamma_net(z)
        beta = self.beta_net(z)
        res = (1 + self.prior_gamma_reg_L2 * gamma) * phis + (self.prior_beta_reg_L2 * beta)
        return res


class TensorFactorisation(torch.nn.Module):
    def __init__(self, input_size, output_size, nb_matrix=10):
        super(TensorFactorisation, self).__init__()
        self.weights_net = Linear(input_size, nb_matrix)
        self.matrices = Parameter(
            torch.Tensor(nb_matrix, output_size, output_size))
        stdv = 1. / np.sqrt(nb_matrix)
        self.matrices.data.uniform_(-stdv, stdv)

    def forward(self, phis, z):
        new_weights = self.weights_net(z)[0]
        new_weights /= new_weights.sum()
        new_weights = new_weights.view(new_weights.size(0), 1, 1).expand(*self.matrices.size())
        matrix = torch.sum(self.matrices * new_weights, dim=0)
        # matrix = torch.sum(torch.stack([self.matrices[i] * w for i, w in enumerate(new_weights)]), dim=0)
        res = linear(phis, matrix)
        return res


class LateFusion(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=None):
        super(LateFusion, self).__init__()
        hidden_sizes = [] if hidden_sizes is None else hidden_sizes
        fusion_layers, inp_size = [], input_size + output_size
        for out_size in list(hidden_sizes) + [output_size]:
            fusion_layers.append(Linear(inp_size, out_size))
            fusion_layers.append(ReLU())
            inp_size = out_size
        self.fusion_net = Sequential(*fusion_layers)
        print(self.fusion_net)

    def forward(self, phis, z):
        new_inputs = torch.cat((phis, z.expand(phis.size(0), z.size(1))), dim=1)
        res = self.fusion_net(new_inputs)
        return res


class FeatureExtractorConditioner(torch.nn.Module):
    def __init__(self, feature_extractor, task_descr_encoder=None, use_task_var=False, use_data_encoder=True,
                 arch='EW', nb_matrix=2, hidden_sizes=None):
        super(FeatureExtractorConditioner, self).__init__()
        assert (task_descr_encoder is not None) or use_data_encoder, \
            "Specify the task_descr_encoder or set use_data_encoder to True"
        hidden_sizes = [] if hidden_sizes is None else hidden_sizes
        self.feature_extractor = feature_extractor
        self.task_descr_encoder = task_descr_encoder
        self.use_task_var = use_task_var
        self.use_data_encoder = use_data_encoder

        self.task_repr_size = 0
        if self.use_data_encoder:
            data_encoder_size = int(self.feature_extractor.output_dim / 2)
            self.data_encoder = Sequential(
                Linear(self.feature_extractor.output_dim, data_encoder_size),
                Tanh(),
                Linear(data_encoder_size, data_encoder_size))
            self.task_repr_size += data_encoder_size * 2

        if self.task_descr_encoder is not None:
            self.task_repr_size += self.task_descr_encoder.output_dim

        if arch.lower() in ['ew', 'film', 'elementwise']:
            self.conditioner = Film(self.task_repr_size, self.feature_extractor.output_dim)
        elif arch.lower() in ['tf', 'tensorfacorisation']:
            self.conditioner = TensorFactorisation(self.task_repr_size, self.feature_extractor.output_dim,
                                                   nb_matrix)
        elif arch.lower() in ['lf', 'latefusion']:
            self.conditioner = LateFusion(self.task_repr_size, self.feature_extractor.output_dim,
                                          hidden_sizes)
        else:
            raise Exception('arch {} is not allowed. The value should be: ew, tf, lf ')

        if self.use_task_var:
            self.task_encoder_mu_net = Sequential(Linear(self.task_repr_size, self.task_repr_size),
                                                  Hardtanh(min_val=-1, max_val=1))
            self.task_encoder_logvar_net = Sequential(Linear(self.task_repr_size, self.task_repr_size),
                                                      Hardtanh(min_val=-9, max_val=-2))

    def compute_task_mu_logvar(self, task_descr, inputs=None, targets=None):
        z = []
        if self.use_data_encoder or (task_descr is None):
            z.append(self.compute_data_repr(inputs, targets).unsqueeze(dim=0))
        if self.task_descr_encoder is not None and (task_descr is not None):
            z.append(self.task_descr_encoder(task_descr.unsqueeze(dim=0)))
        if len(z) == 0:
            raise Exception('Please provide the task_descr or the inputs and targets of the episode')

        z = torch.cat(z, dim=1)

        if self.use_task_var:
            mu_task = self.task_encoder_mu_net(z)
            logvar_task = self.task_encoder_logvar_net(z)
            return mu_task, logvar_task
        else:
            return z, None

    def compute_data_repr(self, inputs, targets):
        phis = self.feature_extractor(inputs)
        x = targets * phis
        y = self.data_encoder(x)
        return torch.cat((torch.mean(y, dim=0), torch.std(y, dim=0)), dim=0)

    def forward(self, inputs, mu_task, logvar_task=None):
        phis = self.feature_extractor(inputs)
        z = reparameterize(mu_task, logvar_task, self.training and self.use_task_var)
        res = self.conditioner(phis, z)
        return res
