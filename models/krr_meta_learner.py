import torch
from torch.nn import MSELoss
from torch.autograd import Variable
from torch.nn import Parameter, Dropout
from torch.optim import Adam
from torch.nn.functional import normalize
from pytoune.utils import variables_to_tensors, tensors_to_variables
from .base import Model, MetaLearnerRegression
from .modules import *

HARDTANH_LIMIT = 5


def compute_kernel(x, y, kernel, gamma=1):
    if kernel.lower() == 'linear':
        K = torch.mm(x, y.t())
    elif kernel.lower() == 'rbf':
        x_i = x.unsqueeze(1)
        y_j = y.unsqueeze(0)
        xmy = ((x_i - y_j) ** 2).sum(2)
        K = torch.exp(-gamma * xmy)
    # elif kernel.lower() == 'poly':
    #     p = 2
    #     K = torch.pow(1 + torch.mm(x, y.t()), p)
    else:
        K = torch.mm(x, y.t())
    return K


class KrrLearner(torch.nn.Module):
    def __init__(self, l2_penalty, kernel='linear', center_kernel=False, gamma=1):
        super(KrrLearner, self).__init__()
        self.l2_penalty = l2_penalty
        self.alpha = None
        self.phis_train = None
        self.kernel = kernel
        self.center_kernel = center_kernel
        self.gamma = gamma

    def fit(self, phis, y):
        batch_size_train = phis.size(0)
        K = compute_kernel(phis, phis, self.kernel, self.gamma)
        I = torch.eye(batch_size_train)
        if torch.cuda.is_available():
            I = I.cuda()
        I = Variable(I)
        if self.center_kernel:
            self.H = I - (1/batch_size_train)
            K = torch.mm(torch.mm(self.H, K), self.H)
            self.y_mean = torch.mean(y)
            self.K = K
        else:
            self.y_mean = 0

        tmp = torch.inverse(K + self.l2_penalty * I)
        self.alpha = torch.mm(tmp, (y-self.y_mean))
        self.phis_train = phis
        self.w_norm = torch.norm(torch.mm(self.alpha.t(), self.phis_train), p=2)

    def forward(self, phis):
        K = compute_kernel(phis, self.phis_train, self.kernel)
        if self.center_kernel:
            K_mean = torch.mean(self.K, dim=1)
            K = torch.mm(K - K_mean, self.H)
            y = torch.mm(K, self.alpha) + self.y_mean
        else:
            y = torch.mm(K, self.alpha)
        return y


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
    def __init__(self, feature_extractor, l2_mode='constant',
                 kernel='linear', center_kernel=False, gamma=1,
                 initial_l2=0.1, y_scaling_factor=1, constrain_phi_norm=False):
        """
        In the constructor we instantiate an lstm module
        """
        super(KrrMetaNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.l2_mode = l2_mode
        self.kernel = kernel
        self.center_kernel = center_kernel
        self.initial_l2 = initial_l2
        self.y_scaling_factor = y_scaling_factor
        self.constrain_phi_norm = constrain_phi_norm

        if l2_mode == 'variable':
            self.l2 = L2Net(self.feature_extractor.output_dim, 1)
        elif l2_mode == 'unique':
            self.l2 = Parameter(torch.FloatTensor([initial_l2]), requires_grad=True)
            self.gamma = Variable(torch.FloatTensor([gamma]), requires_grad=True)
        else:
            self.l2 = Variable(torch.FloatTensor([initial_l2]), requires_grad=False)
            self.gamma = Variable(torch.FloatTensor([gamma]), requires_grad=False)
            if torch.cuda.is_available():
                self.l2 = self.l2.cuda()
                self.gamma = self.gamma.cuda()
        self.attr_watched = dict(l2=0, w_norm=0, phi_norm=0)

    def __forward(self, episode):
        x_train, y_train = episode['Dtrain']
        y_train = y_train * self.y_scaling_factor
        phis = self.feature_extractor(x_train)
        if self.l2_mode == 'variable':
            l2 = self.l2((phis, y_train))
        else:
            l2 = self.l2
            gamma = self.gamma
            l2 = torch.nn.Hardtanh(0.0001, 100)(l2)
            gamma = torch.nn.Hardtanh(0.0001, 100)(gamma)
        if self.l2_mode == 'best_theorique' or self.l2_mode == 'best_mario':
            phis = normalize(phis)

        learner = KrrLearner(l2, self.kernel, self.center_kernel, gamma=gamma)
        learner.fit(phis, y_train)

        self.attr_watched['l2'] = l2.data.cpu().numpy()[0]
        self.attr_watched['gamma'] = gamma.data.cpu().numpy()[0]
        self.attr_watched['phi_norm'] = torch.mean(phis.norm(p=2, dim=1)).data.cpu().numpy()[0]
        self.attr_watched['w_norm'] = learner.w_norm.data.cpu().numpy()[0]
        self.attr_watched['y_mean'] = torch.mean(y_train).data.cpu().numpy()[0]
        self.attr_watched['y_std'] = torch.std(y_train).data.cpu().numpy()[0]

        x_test = episode['Dtest']
        n = x_test.size(0)
        batch_size = 512
        if n > batch_size:
            outs = [learner(self.feature_extractor(x_test[i:i+batch_size]))
                    for i in range(0, n, batch_size)]
            res = torch.cat(outs)
        else:
            res = learner(self.feature_extractor(x_test))
        y_pred = res * (1/self.y_scaling_factor)
        return y_pred

    def forward(self, episodes):
        return [self.__forward(episode) for episode in episodes]


class KrrMetaLearner(MetaLearnerRegression):

    def __init__(self, feature_extractor, l2_mode, lr=0.001,
                 kernel='linear', center_kernel=False, gamma=1,
                 initial_l2=0.1, y_scaling_factor=1, constrain_phi_norm=False):
        super(KrrMetaLearner, self).__init__()
        self.l2_mode = l2_mode
        self.lr = lr
        self.network = KrrMetaNetwork(feature_extractor, l2_mode,
                                      kernel, center_kernel, gamma,
                                      initial_l2, y_scaling_factor, constrain_phi_norm)
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
