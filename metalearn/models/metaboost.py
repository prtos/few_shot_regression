import torch
import math
from torch.nn import MSELoss, Linear, Parameter
from torch.optim import Adam
from pytoune.framework import Model
from .base import MetaLearnerRegression, FeaturesExtractorFactory, MetaNetwork
from .krr import KrrLearner
from .utils import reset_BN_stats


class MetaBoostNetwork(MetaNetwork):
    def __init__(self, feature_extractor_params, n_estimators, l2=0.1):
        """
        In the constructor we instantiate an lstm module
        """
        super(MetaBoostNetwork, self).__init__()
        self.feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
        self.voters = Parameter(torch.Tensor(self.feature_extractor.output_dim, n_estimators))
        stdv = 1. / math.sqrt(n_estimators)
        self.voters.data.uniform_(-stdv, stdv)

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
        x_vote = torch.mm(phis, self.voters)
        learner = KrrLearner(self.l2, dual=False)
        learner.fit(x_vote, y_train)
        # voters_w = learner.w
        topk_idx = torch.topk(learner.w, k=10, dim=0)[1]
        # voters_w = torch.zeros_like(learner.w).scatter(0, topk_idx, learner.w)
        voters = self.voters[:, topk_idx.view(-1)]
        vw = torch.gather(learner.w, 0, topk_idx)

        def eval(X):
            return torch.mm(torch.mm(self.feature_extractor(X), voters), vw)
        # Testing part of the episode
        self.feature_extractor.eval()
        x_test, _ = episode['Dtest']
        n, batch_size = len(x_test), 64
        res = torch.cat([eval(x_test[i:i+batch_size]) for i in range(0, n, batch_size)])
        return res

    def forward(self, episodes):
        return [self.__forward(episode) for episode in episodes]


class MetaBoostLearner(MetaLearnerRegression):

    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, **kwargs):
        network = MetaBoostNetwork(*args, **kwargs)
        super(MetaBoostLearner, self).__init__(network, optimizer, lr, weight_decay)