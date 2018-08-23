import torch
import numpy as np
from torch.nn import Parameter
from torch.nn.functional import mse_loss, softmax, log_softmax, nll_loss
from torch.optim import Adam
from pytoune.framework import Model
from .base import MetaLearnerRegression
from few_shot_regression.utils.feature_extraction import ClonableModule
from few_shot_regression.utils.metric import vse
from .krr import *
from .conditioning import FeatureExtractorConditioner
from .utils import KL_div_diag_multivar_normals, sigm_heating, normal_entropy

# debug in command-line with: import pdb; pdb.set_trace()


class MetaKrrMultiKernelsNetwork(torch.nn.Module):
    def __init__(self, feature_extractor: ClonableModule, task_descr_extractor=None, conditioner_params=None,
                 use_task_var=False, use_data_encoder=True, beta_kl=1.0, l2=0.1):
        super(MetaKrrMultiKernelsNetwork, self).__init__()
        conditioner_params = {} if conditioner_params is None else conditioner_params
        self.feature_extractor = feature_extractor
        self.task_descr_extractor = task_descr_extractor
        self.step = 0
        self.writer = None
        self.beta_kl = beta_kl

        self.l2 = torch.FloatTensor([l2])
        self.l2.requires_grad_(False)
        if torch.cuda.is_available():
            self.l2 = self.l2.cuda()

        self.use_task_var = use_task_var
        self.use_data_encoder =use_data_encoder
        self.conditionned_feat_extract = FeatureExtractorConditioner(self.feature_extractor, self.task_descr_extractor,
                                                                     **conditioner_params, use_task_var=use_task_var,
                                                                     use_data_encoder=self.use_data_encoder)
        if self.use_task_var:
            # self.prior_mu = Parameter(torch.zeros(self.conditionned_feat_extract.task_repr_size))
            # self.prior_log_var = Parameter(torch.zeros(self.conditionned_feat_extract.task_repr_size))
            self.prior_mu = torch.zeros(self.conditionned_feat_extract.task_repr_size)
            self.prior_log_var = torch.zeros(self.conditionned_feat_extract.task_repr_size)
            self.prior_mu.requires_grad_(False)
            self.prior_log_var.requires_grad_(False)
            if torch.cuda.is_available():
                self.prior_mu = self.prior_mu.cuda()
                self.prior_log_var = self.prior_log_var.cuda()

        self.meta_training = True
        # self.step_log = dict()

    def set_writer(self, writer):
        self.writer = writer

    def __forward(self, episode):
        task_descr = episode['task_descr']

        if self.meta_training:
            loss = 0
            for (x_train, y_train), (x_test, y_test) in [(episode['Dtrain'], episode['Dtest']),
                                                         (episode['Dtest'], episode['Dtrain'])]:
                mu, log_var = self.conditionned_feat_extract.compute_task_mu_logvar(task_descr, x_train, y_train)
                self.mu_tau.append(mu)
                if self.use_task_var:
                    kl = KL_div_diag_multivar_normals(mu, log_var, self.prior_mu, self.prior_log_var)
                    p = sigm_heating(self.step, self.beta_kl)
                else:
                    kl, p = 0, 0
                self.conditionned_feat_extract.train()
                phis = self.conditionned_feat_extract(x_train, mu, log_var)
                learner = KrrLearner(self.l2)
                try:
                    learner.fit(phis, y_train)
                except:
                    print('Inversion Error')
                    print(x_train, y_train)
                    exit()
                self.conditionned_feat_extract.eval()
                y_pred = learner(self.conditionned_feat_extract(x_test, mu, log_var))
                mse = mse_loss(y_pred, y_test)
                loss += mse - p * kl
                self.reg_loss += mse/2
                self.kl_loss += kl/2

            return loss/2
        else:
            x_train, y_train = episode['Dtrain']
            x_test, _ = episode['Dtest']
            n, batch_size = x_test.size(0), 16
            mu, log_var = self.conditionned_feat_extract.compute_task_mu_logvar(task_descr, x_train, y_train)
            n_sampling = 10 if self.use_task_var else 1
            res = []
            for _ in range(n_sampling):
                # training of the episode
                self.conditionned_feat_extract.train()
                phis = self.conditionned_feat_extract(x_train, mu, log_var)
                learner = KrrLearner(self.l2)
                learner.fit(phis, y_train)

                # Testing of the episode
                self.conditionned_feat_extract.eval()
                y_pred = torch.cat([learner(self.conditionned_feat_extract(x_test[i:i+batch_size], mu, log_var))
                                    for i in range(0, n, batch_size)])
                res.append(y_pred)
            y_pred = torch.mean(torch.stack(res), dim=0)
            return y_pred

    def compute_task_repr_loss(self, episodes):
        Nep = len(episodes)
        acc_train, acc_test = [], []
        classes = list(range(Nep))
        classes = torch.Tensor(classes)
        for i, episode in enumerate(episodes):
            acc_train.append(
                self.conditionned_feat_extract.compute_task_mu_logvar(episode['task_descr'], *episode['Dtrain']))
            acc_test.append(
                self.conditionned_feat_extract.compute_task_mu_logvar(episode['task_descr'], *episode['Dtest']))
        score_matrix = torch.zeros((Nep, Nep))
        if torch.cuda.is_available():
            classes = classes.cuda()
            score_matrix = score_matrix.cuda()
        for i in range(Nep):
            for j in range(Nep):
                score_matrix[i, j] = (KL_div_diag_multivar_normals(*acc_train[i], *acc_test[j]) +
                                      KL_div_diag_multivar_normals(*acc_test[j], *acc_train[i]) +
                                      normal_entropy(*acc_train[i]) +
                                      normal_entropy(*acc_test[j]))
        score_matrix = -1 * score_matrix
        classes = classes.to(torch.int64)
        enc_loss = nll_loss(log_softmax(score_matrix, dim=1), classes)
        return enc_loss

    def forward(self, episodes):
        self.reg_loss, self.kl_loss = 0, 0
        self.mu_tau = []
        res = [self.__forward(episode) for episode in episodes]
        N = len(res)
        if self.training:
            self.step += 1
            m = torch.cat(self.mu_tau)
            enc_loss = self.compute_task_repr_loss(episodes) if self.use_data_encoder else 0
            self.data_repr_loss = sigm_heating(self.step, max=self.beta_kl) * enc_loss
            if self.writer is not None:
                scalars = dict(kl_beta=sigm_heating(self.step, self.beta_kl).data.cpu().numpy(),
                               task_repr_beta=sigm_heating(self.step, max=self.beta_kl).data.cpu().numpy(),
                               task_repr_loss=enc_loss if isinstance(enc_loss, int) else enc_loss.data.cpu().numpy(),
                               regr_loss=self.reg_loss if isinstance(self.reg_loss, int) else (self.reg_loss/N).data.cpu().numpy(),
                               kl_loss=self.kl_loss if isinstance(self.kl_loss, float) else (self.kl_loss/N).data.cpu().numpy())
                for k, v in scalars.items():
                    self.writer.add_scalars('others/'+k, {k: v}, self.step)
                self.writer.add_scalars('others/' + 'tau', {'tau': m.mean(dim=0).std().data.cpu().numpy()}, self.step)
                self.writer.add_embedding(m, global_step=self.step)
        return res

    def meta_train(self):
        self.meta_training = True

    def meta_eval(self):
        self.meta_training = False


class MetaKrrMultiKernelsLearner(MetaLearnerRegression):

    def __init__(self, feature_extractor, task_descr_extractor=None, conditioner_params=None, use_task_var=False,
                 use_data_encoder=True, lr=0.001, l2=0.1, beta_kl=1.0):
        super(MetaKrrMultiKernelsLearner, self).__init__()
        self.lr = lr
        self.network = MetaKrrMultiKernelsNetwork(feature_extractor, task_descr_extractor, use_task_var=use_task_var,
                                                  use_data_encoder=use_data_encoder, l2=l2,
                                                  conditioner_params=conditioner_params, beta_kl=beta_kl)
        if torch.cuda.is_available():
            self.network.cuda()
        optimizer = Adam([
                {'params': (p for name, p in self.network.named_parameters() if not name.endswith('reg_L2'))},
                {'params': (p for name, p in self.network.named_parameters() if name.endswith('reg_L2')),
                 'weight_decay': 1e-2}
            ], lr=self.lr)
        self.model = Model(self.network, optimizer, self.metaloss)

    def metaloss(self, y_preds, y_tests):
        if self.network.meta_training:
            res = torch.mean(torch.stack([loss for loss, y_test in zip(y_preds, y_tests)]))
            # print('loss net', res)
            if torch.isnan(res):
                print('y-test', y_tests)
                print(y_tests[0].dtype)
                exit()
            res += self.network.data_repr_loss
        else:
            res = torch.mean(torch.stack([mse_loss(y_pred, y_test) for y_pred, y_test in zip(y_preds, y_tests)]))
        return res

    def fit(self, *args, **kwargs):
        self.network.meta_train()
        return super(MetaKrrMultiKernelsLearner, self).fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        self.network.meta_eval()
        return super(MetaKrrMultiKernelsLearner, self).evaluate(*args, **kwargs)


if __name__ == '__main__':
    pass
