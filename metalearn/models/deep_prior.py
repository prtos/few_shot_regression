import torch, os, json
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import log_softmax, nll_loss, sigmoid, mse_loss
from torch.nn import Linear, Sequential, ReLU, Module, Tanh
from torch.optim import Adam
from tensorboardX import SummaryWriter
from pytoune.framework import Model
from pytoune.framework.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, \
    BestModelRestore, TensorBoardLogger
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
from metalearn.models.perspectron import PerspectronEncoderNetwork, PerspectronEncoderLearner, hloss_from_agreement_matrix
from metalearn.models.utils import to_unit, sigm_heating
from metalearn.models.base import MetaLearnerRegression, FeaturesExtractorFactory, MetaNetwork


# debug in command-line with: import pdb; pdb.set_trace()
def reparameterize(mu, var, n_samples=1, identical_samples=False, is_training=False):
    if is_training and var is not None:
        std = torch.std(var)
        if identical_samples:
            eps = torch.randn(*mu.shape).expand(n_samples, *mu.shape)
        else:
            eps = torch.randn(n_samples, *mu.shape)
        if mu.is_cuda:
            eps = eps.cuda()
        return mu + (std * eps)
    else:
        return mu.expand(n_samples, *mu.shape)


def log_pdf(y, mu, std):
    cov = torch.diag(std.view(-1))
    return MultivariateNormal(mu.view(-1), cov).log_prob(y.view(-1))


def batch_kl(mus, stds, prior_mu=None, prior_std=None):
    assert mus is not None, "mu1 should not be None"
    assert len(mus.shape) == 2
    if prior_mu is None:
        prior_mu = torch.zeros_like(mus[0])
    if prior_std is None:
        prior_std = torch.ones_like(mus[0])
    if stds is None:
        stds = torch.ones_like(mus)

    kl = torch.stack([kl_divergence(MultivariateNormal(mu, torch.diag(std)),
                   MultivariateNormal(prior_mu, torch.diag(prior_std)))
     for mu, std in zip(mus, stds)]).sum()

    return kl


def std_activation(pre_std):
    # return (sigmoid(pre_std) * 0.1) + 1e-3
    return sigmoid(pre_std - 5)*0.05 + 0.001
    # return sigmoid(pre_std)


class ResidualBlock(Module):
    def __init__(self, input_dim, block_depth=2, scale_down_factor=2.):
        super(ResidualBlock, self).__init__()
        intern_dim = int(input_dim / scale_down_factor)
        layers = [Tanh()]
        in_dim, out_dim = input_dim, intern_dim
        for i in range(block_depth -1):
            layers.append(Linear(in_dim, out_dim))
            layers.append(Tanh())
            in_dim, out_dim = out_dim, intern_dim
        layers.append(Linear(in_dim, input_dim))
        self.net = Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)
    

class LateFusion(torch.nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size, nb_layers=1):
        super(LateFusion, self).__init__()
        if nb_layers <= 0:
            self.od = input_size1 + input_size2
            self.fusion_net = None
        else:
            hidden_sizes = [hidden_size]*nb_layers
            first_projection = Linear(input_size1 + input_size2, hidden_size)
            fusion_layers = [first_projection]
            for out_size in hidden_sizes:
                fusion_layers.append(ResidualBlock(out_size))
            self.fusion_net = Sequential(*fusion_layers)
            self.od = hidden_size

    def forward(self, inputs):
        phis, zs = inputs
        new_inputs = torch.cat((phis, zs), dim=1)
        res = new_inputs if self.fusion_net is None else self.fusion_net(new_inputs)
        return res

    @property
    def output_dim(self):
        return self.od


class DeepPriorNetwork(MetaNetwork):
    def __init__(self, input_dim=None, feature_extractor_params=None, 
                 task_encoder_params=None,
                 fusion_layer_size=100, fusion_nb_layer=0, 
                 beta_kl=1.0, task_repr_dim=None):
        super(DeepPriorNetwork, self).__init__()
        if feature_extractor_params is None:
            self.feature_extractor = None
        else:
            self.feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
            input_size = self.feature_extractor.output_dim

        if task_encoder_params is None:
            self.task_repr_extractor = None
            assert task_repr_dim is not None, 'task_repr_dim cant be None if task_descr_extractor_params is None'
        else:
            self.task_repr_extractor = PerspectronEncoderNetwork(self.feature_extractor, 
                                        is_latent_discrete=False, **task_encoder_params)
            task_repr_dim = self.task_repr_extractor.output_dim

        fnet = LateFusion(input_size, task_repr_dim, fusion_layer_size, nb_layers=fusion_nb_layer)
        out_layer = Linear(fnet.output_dim, 2)
        self.fusion_net = Sequential(fnet, out_layer)
        self.beta_kl = beta_kl
        self.is_eval = True
    
    @property
    def return_var(self):
        return True

    def _forward(self, episode, task_repr_params, z_sampling=False, z_identical_samples=False):
        x_test, _ = episode['Dtest']
        zs = reparameterize(*task_repr_params, n_samples=x_test.shape[0],
            identical_samples=z_identical_samples, is_training=(self.training or z_sampling))
        phis = x_test if self.feature_extractor is None else self.feature_extractor(x_test)
        outs = self.fusion_net((phis, zs))
        y_mean, y_std = torch.split(outs, 1, dim=1)
        y_std = std_activation(y_std)
        return y_mean, y_std

    def forward(self, episodes):
        if self.task_repr_extractor:
            tasks_repr_params = self.task_repr_extractor(episodes)
        else:
            temp = torch.stack([ep['task_descr'] for ep in episodes])
            tasks_repr_params = (temp, torch.zeros_like(temp)), (temp, torch.zeros_like(temp))
        tasks_repr_params_train, _ = tasks_repr_params

        # task
        # l_tests = [episode['Dtest'][0].shape[0] for episode in episodes]
        # x_tests = torch.cat([episode['Dtest'][0] for episode in episodes], dim=0)
        # zs = torch.torch.cat([reparameterize(*mu_std, n_samples=l_tests[i], is_training=self.training)
        #                       for i, mu_std in enumerate(zip(*tasks_repr_params_train))])
        # phis = x_tests if self.feature_extractor is None else self.feature_extractor(x_tests)

        # outs = self.fusion_net((phis, zs))
        # y_mean, y_std = torch.split(outs, 1, dim=1)
        # y_std = std_activation(y_std)
        # res = list(zip(torch.split(y_mean, l_tests, dim=0), torch.split(y_std, l_tests, dim=0)))

        res = [self._forward(episode, task_repr_params)
               for episode, task_repr_params in zip(episodes, zip(*tasks_repr_params_train))]
        if self.is_eval:   
            means, stds = list(zip(*res))
            # return (torch.stack(means, dim=0), torch.stack(stds, dim=0))
            return res
        else:
            return res, tasks_repr_params


class DeepPriorLearner(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, cotraining=False, pretraining=False, **kwargs):
        network = DeepPriorNetwork(*args, **kwargs)
        super(DeepPriorLearner, self).__init__(network, optimizer, lr, weight_decay)
        self.lr = lr
        self.cotraining = cotraining
        self.pretraining = pretraining

    def _compute_aux_return_loss(self, y_preds, y_tests):
        y_preds_per_task, tasks_repr_params = y_preds
        tasks_repr_params_train = tasks_repr_params[0]

        if self.model.task_repr_extractor:
            ag_matrix = self.model.task_repr_extractor.compute_agreement_matrix(*tasks_repr_params)
            prior = self.model.task_repr_extractor.get_prior()
            kl = batch_kl(*tasks_repr_params_train, *prior).mean(dim=-1).mean(dim=-1)
        else:
            ag_matrix = torch.eye(len(y_tests))
            prior = None
            kl = 0

        lml = torch.mean(torch.stack([log_pdf(y_test.view(-1), y_pred[0].view(-1), y_pred[1].view(-1))
                         for y_pred, y_test in zip(y_preds_per_task, y_tests)]))
        mse = torch.mean(torch.stack([mse_loss(y_pred[0].view(-1), y_test.view(-1))
                         for y_pred, y_test in zip(y_preds_per_task, y_tests)]))
        
        norm_std = torch.mean(torch.stack([torch.norm(std)**2 for (_, std) in y_preds_per_task]))

        hloss = hloss_from_agreement_matrix(ag_matrix)
        kl_weight = sigm_heating(self.train_step, self.model.beta_kl, 30000) if self.model.training else 0
        kl_weight = torch.Tensor([kl])[0]
        loss = -lml + kl_weight*kl      # + norm_std
        if self.cotraining:
            loss = loss + hloss

        scalars = dict(encoder_milbo=-1*hloss,
                        kl_value=kl,
                        neg_log_marginal_likelihood=-lml,
                        mse=mse,
                        kl_weight=kl_weight,
                        norm_std=norm_std)
                        
        return loss, scalars


    def fit(self, *args, **kwargs):
        if self.pretraining:
            p = PerspectronEncoderLearner(network=self.model.task_repr_extractor, lr=self.lr)
            p.fit(*args, **kwargs)
        super(DeepPriorLearner, self).fit(*args, **kwargs)

if __name__ == '__main__':
    pass
