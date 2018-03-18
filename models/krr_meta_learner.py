import torch
from torch.nn import MSELoss
from torch.autograd import Variable
from torch.nn import Parameter
from torch.optim import Adam
from pytoune.utils import variables_to_tensors, tensors_to_variables
from .base import Model, MetaLearnerRegression
from .modules import *

HARDTANH_LIMIT = 5


def compute_kernel(x, y, kernel):
    if kernel.lower() == 'linear':
        K = torch.mm(x, y.t())
    # elif kernel.lower() == 'rbf':
    #     gamma = 1 / 2
    #     x_i = x.unsqueeze(1)
    #     y_j = y.unsqueeze(0)
    #     xmy = ((x_i - y_j) ** 2).sum(2)
    #     K = torch.exp(-gamma * xmy)
    # elif kernel.lower() == 'poly':
    #     p = 2
    #     K = torch.pow(1 + torch.mm(x, y.t()), p)
    else:
        K = torch.mm(x, y.t())
    return K


class KrrLearner(torch.nn.Module):

    def __init__(self, l2_penalty, kernel='linear'):
        super(KrrLearner, self).__init__()
        self.l2_penalty = l2_penalty
        self.alpha = None
        self.phis_train = None
        self.kernel = kernel

    def fit(self, phis, y):
        batch_size_train = phis.size(0)
        K = compute_kernel(phis, phis, self.kernel)
        I = torch.eye(batch_size_train)
        if torch.cuda.is_available():
            I = I.cuda()
        tmp = torch.inverse(K + self.l2_penalty * Variable(I))
        self.alpha = torch.mm(tmp, y)
        self.phis_train = phis

    def forward(self, phis):
        K = compute_kernel(phis, self.phis_train, self.kernel)
        return torch.mm(K, self.alpha)


class L2Net(torch.nn.Module):
    def __init__(self, dim_inputs, dim_targets):
        super(L2Net, self).__init__()
        n_in = (dim_inputs + dim_targets)*4
        self.net = Sequential(
            # GaussianDropout(alpha=0.1),
            Linear(n_in, 1))

    def forward(self, inputs):
        phis, y = inputs
        var = [phis.mean(dim=0), phis.std(dim=0), phis.max(dim=0)[0], phis.min(dim=0)[0],
                       y.mean(dim=0), y.std(dim=0), y.max(dim=0)[0], y.min(dim=0)[0]
               ]
        x = torch.cat(var, 0)
        return self.net(x)


class KrrMetaNetwork(torch.nn.Module):
    def __init__(self, feature_extractor, unique_l2=False, kernel='linear'):
        """
        In the constructor we instantiate an lstm module
        """
        super(KrrMetaNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.unique_l2 = unique_l2
        self.kernel = kernel
        if unique_l2:
            self.l2_params = Parameter(torch.FloatTensor([1e-2]), requires_grad=True)
        else:
            self.l2_network = L2Net(self.feature_extractor.output_dim, 1)

        self.l2_krr = 0

    def __forward(self, episode):
        x_train, y_train = episode['Dtrain']
        phis = self.feature_extractor(x_train)
        if not self.unique_l2:
            l2 = self.l2_network((phis, y_train))
        else:
            l2 = self.l2_params
        l2 = torch.exp(torch.nn.Hardtanh(-HARDTANH_LIMIT, HARDTANH_LIMIT)(l2))
        self.l2_krr = l2.data.cpu().numpy()[0]

        learner = KrrLearner(l2, self.kernel)
        learner.fit(phis, y_train)

        y_pred = learner(self.feature_extractor(episode['Dtest']))
        return y_pred

    def forward(self, episodes):
        return [self.__forward(episode) for episode in episodes]


class KrrMetaLearner(MetaLearnerRegression):

    def __init__(self, feature_extractor, unique_l2, lr=0.001, kernel='linear'):
        super(KrrMetaLearner, self).__init__()
        self.unique_l2 = unique_l2
        self.lr = lr
        self.network = KrrMetaNetwork(feature_extractor, unique_l2, kernel)
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     # dim = 0 [33, xxx] -> [11, ...], [11, ...], [11, ...] on 3 GPUs
        #     self.network = torch.nn.DataParallel(self.network)

        if torch.cuda.is_available():
            self.network.cuda()
        self.loss = MSELoss()
        optimizer = Adam(self.network.parameters(), lr=self.lr)
        self.model = Model(self.network, optimizer, self.metaloss)


if __name__ == '__main__':
    pass
