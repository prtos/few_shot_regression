from torch.nn import MSELoss, Linear, Sequential
from torch.optim import Adam
from torch.nn.functional import normalize, mse_loss
from pytoune.framework import Model
from metalearn.models.base import MetaLearnerRegression
from metalearn.models.krr import *


class MaskedMSE(MSELoss):
    def __init__(self, size_average=True, reduce=True):
        super(MSELoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target_and_mask):
        target, mask = target_and_mask
        non_zeros = torch.nonzero(mask)
        y_true = target[non_zeros[:, 0], non_zeros[:, 1]]
        y_pred = input[non_zeros[:, 0], non_zeros[:, 1]]
        return mse_loss(y_pred, y_true, size_average=self.size_average, reduce=self.reduce)


class MultiTaskNetwork(torch.nn.Module):
    def __init__(self, feature_extractor, ntasks, l2=0.1):
        """
        In the constructor we instantiate an lstm module
        """
        super(MultiTaskNetwork, self).__init__()

        self.feature_extractor = feature_extractor
        self.ntasks = ntasks
        self.l2 = l2

        self.top_layers = Linear(self.feature_extractor.output_dim, ntasks)
        self.net = Sequential(self.feature_extractor, self.top_layers)
        self.phase = 0

    def __forward_test(self, episode):
        x_train, y_train = episode['Dtrain']
        phis = self.feature_extractor(x_train)

        learner = KrrLearner(self.l2)
        learner.fit(phis, y_train)

        x_test = episode['Dtest']
        n = x_test.size(0)
        batch_size = 512
        if n > batch_size:
            outs = [learner(self.feature_extractor(x_test[i:i+batch_size])) for i in range(0, n, batch_size)]
            res = torch.cat(outs)
        else:
            res = learner(self.feature_extractor(x_test))
        return res

    def forward(self, inputs):
        if self.phase == 0:
            return self.net(inputs)
        else:
            return [self.__forward_test(episode) for episode in inputs]


class MultiTaskLearner(MetaLearnerRegression):

    def __init__(self, feature_extractor, ntasks, l2, lr=0.001):
        super(MultiTaskLearner, self).__init__()
        self.lr = lr
        self.network = MultiTaskNetwork(feature_extractor, ntasks, l2)
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     # dim = 0 [33, xxx] -> [11, ...], [11, ...], [11, ...] on 3 GPUs
        #     self.network = torch.nn.DataParallel(self.network)

        if torch.cuda.is_available():
            self.network.cuda()
        self.loss = MaskedMSE()
        optimizer = Adam(self.network.parameters(), lr=self.lr)
        self.model = Model(self.network, optimizer, self.loss)

    def train_test_split(self, dataset, test_size):
        return dataset.train_test_split_for_multitask(test_size=test_size)

    def fit(self, metatrain, valid_size=0.25, n_epochs=100, steps_per_epoch=100,
            max_episodes=None, batch_size=32, log_filename=None, checkpoint_filename=None):
        self.network.phase = 0
        return super(MultiTaskLearner, self).fit(metatrain, valid_size, n_epochs, steps_per_epoch,
                                                 max_episodes, batch_size, log_filename, checkpoint_filename)

    def evaluate(self, metatest):
        self.network.phase = 1
        return super(MultiTaskLearner, self).evaluate(metatest)


if __name__ == '__main__':
    pass
