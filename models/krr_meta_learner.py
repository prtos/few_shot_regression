import torch
from torch.nn import MSELoss
from torch.autograd import Variable
from torch.nn import Parameter
from torch.optim import Adam
from pytoune.utils import variables_to_tensors, tensors_to_variables
from .base import MetaModel, MetaLearnerRegression
from .modules import *
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


class KrrLearner(torch.nn.Module):

    def __init__(self, l2_penalty):
        super(KrrLearner, self).__init__()
        self.l2_penalty = l2_penalty
        self.alpha = None
        self.phis_train = None

    def fit(self, phis, y):
        batch_size_train = phis.size(0)
        K = torch.mm(phis, phis.t())
        I = torch.eye(batch_size_train)
        if torch.cuda.is_available():
            I = I.cuda()
        tmp = torch.inverse(K + self.l2_penalty * Variable(I))
        self.alpha = torch.mm(tmp, y)
        self.phis_train = phis

    def forward(self, phis):
        K = torch.mm(phis, self.phis_train.t())
        return torch.mm(K, self.alpha)


class L2Net(torch.nn.Module):
    def __init__(self, dim_inputs, dim_targets):
        super(L2Net, self).__init__()
        n_in = (dim_inputs + dim_targets)*4
        self.net = Sequential(
            # GaussianDropout(alpha=0.1),
            Linear(n_in, 1), torch.nn.Hardtanh(-5, 5))

    def forward(self, inputs):
        phis, y = inputs
        var = [phis.mean(dim=0), phis.std(dim=0), phis.max(dim=0)[0], phis.min(dim=0)[0],
                       y.mean(dim=0), y.std(dim=0), y.max(dim=0)[0], y.min(dim=0)[0]
               ]
        x = torch.cat(var, 0)
        return torch.exp(self.net(x))


class KrrMetaNetwork(torch.nn.Module):
    def __init__(self, feature_extractor, unique_l2=False):
        """
        In the constructor we instantiate an lstm module
        """
        super(KrrMetaNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.unique_l2 = unique_l2
        if unique_l2:
            self.l2 = Parameter(torch.FloatTensor([1e-2]), requires_grad=True)
        else:
            self.l2_network = L2Net(self.feature_extractor.output_dim, 1)

    def __forward(self, episode):
        x_train, y_train = episode['Dtrain']
        phis = self.feature_extractor(x_train)
        if self.unique_l2:
            l2 = torch.nn.Hardtanh(-5, 5)(self.l2)
        else:
            l2 = self.l2_network((phis, y_train))
        learner = KrrLearner(l2)
        learner.fit(phis, y_train)
        return learner

    def forward(self, episodes):
        return [self.__forward(episode) for episode in episodes]


class KrrMetaLearner(MetaLearnerRegression):

    def __init__(self, feature_extractor, unique_l2, lr=0.001):
        super(KrrMetaLearner, self).__init__()
        self.unique_l2 = unique_l2
        self.lr = lr
        self.network = KrrMetaNetwork(feature_extractor, unique_l2)
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     # dim = 0 [33, xxx] -> [11, ...], [11, ...], [11, ...] on 3 GPUs
        #     self.network = torch.nn.DataParallel(self.network)

        if torch.cuda.is_available():
            self.network.cuda()
        self.loss = MSELoss()
        optimizer = Adam(self.network.parameters(), lr=self.lr)
        self.model = MetaModel(self.network, optimizer, self.metaloss)

    def metaloss(self, episodes, learners):
        # todo: change the way you handles the loss
        for i, (episode, learner) in enumerate(zip(episodes, learners)):
            x_test, y_test = episode['Dtest']
            phis_test = self.network.feature_extractor(x_test)

            y_pred = learner(phis_test)
            if i == 0:
                loss = self.loss(y_pred, y_test)
            else:
                loss += self.loss(y_pred, y_test)

        return loss/len(episodes)

    def evaluate(self, metatest):
        scores_r2, scores_pcc, sizes = dict(), dict(), dict()
        for batch in metatest:
            batch = tensors_to_variables(batch, volatile=True)
            learners = self.model.predict(batch)
            for episode, learner in zip(batch, learners):
                x_test, y_test = episode['Dtest']
                phis_test = self.network.feature_extractor(x_test)
                y_pred = learner(phis_test)
                x, y = y_test.data.cpu().numpy().flatten(), y_pred.data.cpu().numpy().flatten()
                r2 = float(r2_score(x, y))
                pcc = float(pearsonr(x, y)[0])
                ep_name = "".join([chr(i) for i in episode['name'].data.cpu().numpy()])
                if ep_name in scores_pcc:
                    scores_pcc[ep_name].append(pcc)
                    scores_r2[ep_name].append(r2)
                else:
                    scores_pcc[ep_name] = [pcc]
                    scores_r2[ep_name] = [r2]
                sizes[ep_name] = y_test.size(0)

        return scores_r2, scores_pcc, sizes


if __name__ == '__main__':
    pass
