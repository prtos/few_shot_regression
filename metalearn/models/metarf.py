import torch
from torch.nn import MSELoss
from torch.optim import Adam
from pytoune.framework import Model
from .base import MetaLearnerRegression, FeaturesExtractorFactory, MetaNetwork
from .krr import *
from .utils import reset_BN_stats


class RandomForestRegression(torch.nn.Module):
    def __init__(self, l2, n_estimators):
        super(RandomForestRegression, self).__init__()
        self.l2 = l2
        self.alphas = None
        self.phis_train = None
        self.n_estimators = n_estimators
        self.nf_selected = None
        self.trees_features_idx = None

    def fit(self, phis, y):
        k = self.n_estimators
        n_train, n_features = phis.shape
        self.nf_selected = int(n_features / 3.)

        # sampling
        # samples_idx = Multinomial(total_count=n_train).sample(k * n_train)
        samples_idx = torch.multinomial(torch.ones(k, n_train), n_train).view(-1)
        y_trees = y[samples_idx].reshape((k, n_train, -1))
        self.trees_features_idx = torch.multinomial(torch.ones(k, n_features), self.nf_selected)
        temp = self.trees_features_idx.unsqueeze(1).expand(k, n_train, self.nf_selected).reshape(-1, self.nf_selected)
        self.phis_train = torch.gather(phis[samples_idx], dim=1, index=temp).reshape(
            k, n_train, -1)

        # learning
        batch_K = torch.bmm(self.phis_train, self.phis_train.transpose(1, 2))
        I = torch.eye(n_train, device=batch_K.device)
        self.alphas, _ = torch.gesv(y_trees, (batch_K + self.l2*I))
        return self

    def forward(self, phis):
        n = phis.size(0)
        samples_idx = torch.arange(n*self.n_estimators) % n
        temp = self.trees_features_idx.unsqueeze(1).expand(self.n_estimators, n, self.nf_selected).reshape(
            -1, self.nf_selected)
        phis_test = torch.gather(phis[samples_idx], dim=1, index=temp).reshape(self.n_estimators, n, -1)
        batch_K = torch.bmm(phis_test, self.phis_train.transpose(1, 2))
        return torch.bmm(batch_K, self.alphas).mean(dim=0)


class MetaRFNetwork(MetaNetwork):
    def __init__(self, feature_extractor_params, n_estimators_train, l2=0.1, n_estimators_test=None):
        """
        In the constructor we instantiate an lstm module
        """
        super(MetaRFNetwork, self).__init__()
        self.feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
        self.n_estimators_train = n_estimators_train
        self.n_estimators_test = n_estimators_test if n_estimators_test is not None else n_estimators_train

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
        n_estimators = self.n_estimators_train if self.training else self.n_estimators_test
        learner = RandomForestRegression(self.l2, n_estimators=n_estimators)

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


class MetaRFLearner(MetaLearnerRegression):

    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, **kwargs):
        network = MetaRFNetwork(*args, **kwargs)
        super(MetaRFLearner, self).__init__(network, optimizer, lr, weight_decay)

if __name__ == '__main__':
    from time import time
    t0 = time()
    rgr = RandomForestRegression(0.1, 400)
    x_train = torch.rand(10, 1000)
    x_test = torch.rand(7, 1000)
    y = torch.rand(10, 1)
    t1 = time()
    print(t1-t0)
    rgr.fit(x_train, y)
    preds = rgr(x_test)
    t1 = time()
    print(preds.shape, t1-t0)





