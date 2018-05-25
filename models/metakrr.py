import torch
from torch.nn import MSELoss
from torch.autograd import Variable
from torch.nn import Parameter, Dropout
from torch.optim import Adam
from .base import Model, MetaLearnerRegression
from .modules import *
from .krr import *


class L2Net(torch.nn.Module):
    def __init__(self, dim_inputs, dim_targets):
        super(L2Net, self).__init__()
        n_in = (dim_inputs + dim_targets)*4 + dim_inputs
        n = 200
        self.net = Sequential(
            Dropout(0.1),
            Linear(n_in, n),
            ReLU(),
            Linear(n, 1))

    def forward(self, inputs):
        phis, y = inputs
        corr = torch.mm(y.t(), phis)
        var = [phis.mean(dim=0), phis.std(dim=0), phis.max(dim=0)[0], phis.min(dim=0)[0], corr[0],
                       y.mean(dim=0), y.std(dim=0), y.max(dim=0)[0], y.min(dim=0)[0]
               ]
        x = torch.cat(var, 0)
        return torch.nn.Hardtanh(0.01, 10)(torch.exp(self.net(x)))


class KrrMetaNetwork(torch.nn.Module):
    def __init__(self, feature_extractor, kernel='linear', center_kernel=False, l2_init=0.1, gamma_init=1,):
        """
        In the constructor we instantiate an lstm module
        """
        super(KrrMetaNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.kernel = kernel
        self.center_kernel = center_kernel

        self.l2 = Variable(torch.FloatTensor([l2_init]), requires_grad=False)
        self.gamma = Variable(torch.FloatTensor([gamma_init]), requires_grad=False)
        if torch.cuda.is_available():
            self.l2 = self.l2.cuda()
            self.gamma = self.gamma.cuda()
        self.attr_watched = dict(l2=0, w_norm=0, phi_norm=0)

    def __forward(self, episode):
        x_train, y_train = episode['Dtrain']
        phis = self.feature_extractor(x_train)

        learner = KrrLearner(self.l2, self.kernel, self.center_kernel, gamma=self.gamma)
        learner.fit(phis, y_train)

        self.attr_watched['l2'] = self.l2.data.cpu().numpy()[0]
        self.attr_watched['gamma'] = self.gamma.data.cpu().numpy()[0]
        self.attr_watched['phi_norm'] = torch.mean(phis.norm(p=2, dim=1)).data.cpu().numpy()[0]
        self.attr_watched['w_norm'] = learner.w_norm.data.cpu().numpy()[0]

        x_test = episode['Dtest']
        n = x_test.size(0)
        batch_size = 512
        if n > batch_size:
            outs = [learner(self.feature_extractor(x_test[i:i+batch_size])) for i in range(0, n, batch_size)]
            res = torch.cat(outs)
        else:
            res = learner(self.feature_extractor(x_test))
        return res

    def forward(self, episodes):
        return [self.__forward(episode) for episode in episodes]


class KrrMetaLearner(MetaLearnerRegression):

    def __init__(self, feature_extractor, lr=0.001,
                 kernel='linear', center_kernel=False, l2=0.1, gamma=1):
        super(KrrMetaLearner, self).__init__()
        self.lr = lr
        self.network = KrrMetaNetwork(feature_extractor, kernel, center_kernel, l2, gamma)
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
