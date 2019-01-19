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
