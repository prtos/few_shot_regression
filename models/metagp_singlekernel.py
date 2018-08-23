import torch
from torch.nn import MSELoss
from torch.optim import Adam
from pytoune.framework import Model
from .base import MetaLearnerRegression
from .gp import *
from .utils import reset_BN_stats


class MetaGPSingleKernelNetwork(torch.nn.Module):
    def __init__(self, feature_extractor, l2=0.1):
        super(MetaGPSingleKernelNetwork, self).__init__()
        self.feature_extractor = feature_extractor

        self.l2 = torch.FloatTensor([l2])
        self.l2.requires_grad_(False)
        if torch.cuda.is_available():
            self.l2 = self.l2.cuda()
        self.meta_training = False

    def __forward(self, episode):
        x_train, y_train = episode['Dtrain']
        # training part of the episode
        reset_BN_stats(self.feature_extractor)
        if self.meta_training:
            self.feature_extractor.train()
            x_test, y_test = episode['Dtest']
            x_train, y_train = torch.cat([x_train, x_test]), torch.cat([y_train, y_test])
            phis = self.feature_extractor(x_train)
            learner = GPLearner(self.l2)
            learner.fit(phis, y_train)
            return -learner.log_marginal_likelihood()
        else:
            self.feature_extractor.train()
            phis = self.feature_extractor(x_train)
            learner = GPLearner(self.l2)
            learner.fit(phis, y_train)
            # Testing part of the episode
            self.feature_extractor.eval()
            x_test, _ = episode['Dtest']
            n = len(x_test)
            batch_size = 64
            y_pred = torch.cat([learner(self.feature_extractor(x_test[i:i + batch_size]))
                                for i in range(0, n, batch_size)])
            # end of testing
            self.feature_extractor.train()
            return y_pred

    def forward(self, episodes):
        return [self.__forward(episode) for episode in episodes]

    def meta_train(self):
        self.meta_training = True

    def meta_eval(self):
        self.meta_training = False


class MetaGPSingleKernelLearner(MetaLearnerRegression):

    def __init__(self, feature_extractor, lr=0.001, l2=0.1):
        super(MetaGPSingleKernelLearner, self).__init__()
        self.lr = lr
        self.network = MetaGPSingleKernelNetwork(feature_extractor, l2)
        if torch.cuda.is_available():
            self.network.cuda()
        optimizer = Adam(self.network.parameters(), lr=self.lr)
        self.model = Model(self.network, optimizer, self.metaloss)

    def metaloss(self, y_preds, y_tests):
        if self.network.meta_training:
            res = torch.mean(torch.stack([loss for loss, _ in zip(y_preds, y_tests)]))
        else:
            res = torch.mean(torch.stack([mse_loss(y_pred, y_test) for y_pred, y_test in zip(y_preds, y_tests)]))
        return res

    def fit(self, *args, **kwargs):
        self.network.meta_train()
        return super(MetaGPSingleKernelLearner, self).fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        self.network.meta_eval()
        return super(MetaGPSingleKernelLearner, self).evaluate(*args, **kwargs)

if __name__ == '__main__':
    pass
