import torch
from torch.nn import MSELoss
from torch.optim import Adam
from pytoune.framework import Model
from .base import MetaLearnerRegression
from .krr import *
from .utils import reset_BN_stats


class MetaKrrSingleKernelNetwork(torch.nn.Module):
    def __init__(self, feature_extractor, l2=0.1):
        """
        In the constructor we instantiate an lstm module
        """
        super(MetaKrrSingleKernelNetwork, self).__init__()
        self.feature_extractor = feature_extractor

        self.l2 = torch.FloatTensor([l2])
        self.l2.requires_grad_(False)
        if torch.cuda.is_available():
            self.l2 = self.l2.cuda()

    def __forward(self, episode):
        # training part of the episode
        reset_BN_stats(self.feature_extractor)
        self.feature_extractor.train()
        x_train, y_train = episode['Dtrain']
        phis = self.feature_extractor(x_train)


        learner = KrrLearner(self.l2)
        learner.fit(phis, y_train)

        # Testing part of the episode
        self.feature_extractor.eval()
        x_test, _ = episode['Dtest']
        n = len(x_test)
        batch_size = 64
        if n > batch_size:
            outs = [learner(self.feature_extractor(x_test[i:i+batch_size])) for i in range(0, n, batch_size)]
            res = torch.cat(outs)
        else:
            res = learner(self.feature_extractor(x_test))
        # end of testing
        self.feature_extractor.train()
        return res

    def forward(self, episodes):
        return [self.__forward(episode) for episode in episodes]


class MetaKrrSingleKernelLearner(MetaLearnerRegression):

    def __init__(self, feature_extractor, lr=0.001, l2=0.1):
        super(MetaKrrSingleKernelLearner, self).__init__()
        self.lr = lr
        self.network = MetaKrrSingleKernelNetwork(feature_extractor, l2)
        if torch.cuda.is_available():
            self.network.cuda()
        optimizer = Adam(self.network.parameters(), lr=self.lr)
        self.model = Model(self.network, optimizer, self.metaloss)


if __name__ == '__main__':
    pass
