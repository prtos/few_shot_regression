# MARS: Metaleaning through adaptative representation selection

import torch
import math
import numpy as np
from torch.nn import MSELoss, Linear, Parameter, ModuleList
from torch.nn.functional import mse_loss
from torch.optim import Adam
from pytoune.framework import Model
import matplotlib.pyplot as plt
from .base import MetaLearnerRegression, FeaturesExtractorFactory, MetaNetwork, to_unit
from .krr import KrrLearner
from .utils import reset_BN_stats, annealed_softmax


def stack_barplot(matrix):
    plt.figure()
    # Plot bars and create text labels for the table
    cell_text = []
    cum_ = np.concatenate((np.zeros((1, matrix.shape[1])), matrix.cumsum(axis=0)[:-1]), axis=0)
    x = np.arange(matrix.shape[1])
    for i, row in enumerate(matrix):
        plt.bar(x, row, bottom=cum_[i])
    return plt.gcf()


class MarsNetwork(MetaNetwork):
    def __init__(self, feature_extractor_params, n_estimators, l2=0.1, cooling_factor=0.001):
        """
        In the constructor we instantiate an lstm module
        """ 
        super(MarsNetwork, self).__init__()
        self.cooling_factor = cooling_factor
        self.feature_extractors = ModuleList() 
        for i in range(n_estimators):
            self.feature_extractors.append(FeaturesExtractorFactory()(**feature_extractor_params))
        self.n_estimators = n_estimators
        self.l2 = torch.FloatTensor([l2])
        self.l2.requires_grad_(False)
        if torch.cuda.is_available():
            self.l2 = self.l2.cuda()
        self.step = 0
        self.idx_selected_estimators = []

    def __forward(self, episode):
        # training part of the episode
        for fe in self.feature_extractors:
            fe.train()
        x_train, y_train = episode['Dtrain']
        n, k = len(x_train), self.n_estimators
        if self.n_estimators > 1:
            # generate features using the different extractors
            phis = torch.cat([f(x_train) for f in self.feature_extractors], dim=0)
            y = y_train.unsqueeze(0).expand(k, *y_train.shape).reshape(k*n, y_train.shape[-1])
            # print(phis.shape)
            # print(y.shape)

            temp = torch.arange(0, k*n, n).view(-1, 1, 1)
            train_idx = (torch.eye(n, n)==0).nonzero()[:, 1].view(n, n-1)
            train_idx = train_idx.unsqueeze(dim=0).expand(k, *train_idx.shape)
            train_idx = (train_idx + temp).reshape(k*n, n-1)
            test_idx = torch.arange(n).unsqueeze(dim=1)
            test_idx = test_idx.unsqueeze(dim=0).expand(k, *test_idx.shape)
            test_idx = (test_idx + temp).reshape(k*n, 1)
            # print(train_idx.shape)
            # print(test_idx.shape)
            phis_cv_train, phis_cv_test = phis[train_idx], phis[test_idx]
            y_cv_train, y_cv_test = y[train_idx], y[test_idx]
            # print(phis_cv_train.size(), phis_cv_test.size())
            # print(y_cv_train.size(), y_cv_test.size())
            
            # CV learning
            batch_K = torch.bmm(phis_cv_train, phis_cv_train.transpose(1, 2))
            I = torch.eye(n-1, device=batch_K.device)
            alphas, _ = torch.gesv(y_cv_train, (batch_K + self.l2*I))
            # print(alphas.shape)

            # CV testing
            batch_K = torch.bmm(phis_cv_test, phis_cv_train.transpose(1, 2))
            y_cv_preds = torch.bmm(batch_K, alphas)
            # print(y_cv_preds.shape)

            # CV metric computation and selection
            # print(y_cv_preds.shape, y_cv_test.shape)
            cv_losses = ((y_cv_preds - y_cv_test)**2).sum(dim=2).reshape(n, k).sum(dim=0)
            # print(cv_losses.shape)
            if self.training:
                probs = annealed_softmax(-1*cv_losses, t=self.step, 
                            cooling_factor=self.cooling_factor)
            else:
                probs = annealed_softmax(-1*cv_losses, t=self.step, 
                            cooling_factor=(1.0/self.step))
            if torch.any(probs < 0):
                print(probs)
                print('negative probs')
                exit()
            best_fe_idx = torch.multinomial(probs, 1)[0]
            best_fe = self.feature_extractors[best_fe_idx]
            self.idx_selected_estimators.append(best_fe_idx)
        else:
            best_fe = self.feature_extractors[0]
        # retraining
        phis_train = best_fe(x_train)
        learner = KrrLearner(self.l2, dual=False)
        learner.fit(phis_train, y_train)

        # Testing part of the episode
        for fe in self.feature_extractors:
            fe.eval()
        x_test, _ = episode['Dtest']
        n, batch_size = len(x_test), 64
        res = torch.cat([learner(best_fe(x_test[i:i+batch_size])) for i in range(0, n, batch_size)])
        return res

    def forward(self, episodes):
        if self.training:
            self.step += 1
        self.idx_selected_estimators = []
        return [self.__forward(episode) for episode in episodes]


class MarsLearner(MetaLearnerRegression):

    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, **kwargs):
        network = MarsNetwork(*args, **kwargs)
        super(MarsLearner, self).__init__(network, optimizer=optimizer, lr=lr, weight_decay=weight_decay)
        self.probs_fe = []
        self.step = 0

    def _compute_aux_return_loss(self, y_preds, y_tests):
        loss = torch.mean(torch.stack([mse_loss(y_pred, y_test) 
                for y_pred, y_test in zip(y_preds, y_tests)]))
        t = self.step
        tag = 'train' if self.model.training else 'val'
        lr = self.optimizer.param_groups[0]['lr']
        if self.writer is not None:
            scalars = dict(loss=to_unit(loss),
                            lr=lr)
            for k, v in scalars.items():
                self.writer.add_scalar('mars/'+k, v, t)
            if (t == 0) or (t % self.steps_per_epoch != 0):
                w = np.array(self.model.idx_selected_estimators)
                self.probs_fe.append([np.count_nonzero(w == i) 
                        for i in range(self.model.n_estimators)])
            else:
                m = np.array(self.probs_fe).T
                plt.figure()
                bottom = np.zeros(m.shape[1])
                for i, row in enumerate(m):
                    plt.bar(np.arange(m.shape[1]), row, bottom=bottom)
                    bottom += row
                self.writer.add_figure(tag, plt.gcf(), global_step=t)
                self.probs_fe = []
        if torch.isnan(loss).any():
            raise Exception(f'{self.__class__.__name__}: Loss goes NaN')
        self.step += 1
        return loss

