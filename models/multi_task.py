import torch
from torch.nn import MSELoss,
from torch.autograd import Variable
from torch.nn import Parameter, Dropout
from torch.optim import Adam
from torch.nn.functional import normalize
from pytoune.utils import variables_to_tensors, tensors_to_variables
from .base import Model, MetaLearnerRegression
from .modules import *

class MaskedMSE(MSELoss):
    def __init__(self, size_average=True, reduce=True):
        super(MSELoss, self).__init__(size_average)
        self.reduce = reduce

    def forward(self, input, target_mask):
        target, mask = target_mask
        n = torch.sum(mask)

        return F.mse_loss(input[nz[]], target*mask, size_average=self.size_average, reduce=self.reduce)


class KrrLearner(torch.nn.Module):
    def __init__(self, l2_penalty, ):
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
        I = Variable(I)

        tmp = torch.inverse(K + self.l2_penalty * I)
        self.alpha = torch.mm(tmp, y)
        self.phis_train = phis
        self.w_norm = torch.norm(torch.mm(self.alpha.t(), self.phis_train), p=2)

    def forward(self, phis):
        K = torch.mm(phis, self.phis_train.t())
        y = torch.mm(K, self.alpha)
        return y


class MultiTaskNetwork(torch.nn.Module):
    def __init__(self, feature_extractor, ntasks, initial_l2=0.1):
        """
        In the constructor we instantiate an lstm module
        """
        super(MultiTaskNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.ntasks = ntasks
        self.initial_l2 = initial_l2

        self.top_layers = Linear(self.feature_extractor.output_dim, ntasks)
        self.net = Sequential(self.feature_extractor, self.top_layers)
        self.training = 0

    def forward(self, inputs):
        return self.net(inputs)




class MultiTaskLearner(MetaLearnerRegression):

    def __init__(self, feature_extractor, ntasks, l2_mode, lr=0.001):
        super(MultiTaskLearner, self).__init__()
        self.l2_mode = l2_mode
        self.lr = lr
        self.network = MultiTaskNetwork(feature_extractor, ntasks)
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     # dim = 0 [33, xxx] -> [11, ...], [11, ...], [11, ...] on 3 GPUs
        #     self.network = torch.nn.DataParallel(self.network)

        if torch.cuda.is_available():
            self.network.cuda()
        self.loss = MSELoss()
        optimizer = Adam(self.network.parameters(), lr=self.lr)
        self.model = Model(self.network, optimizer, self.metaloss)

    def fit(self, metatrain, metavalid, n_epochs=100, steps_per_epoch=100,
            max_episodes=None, batch_size=32,
            log_filename=None, checkpoint_filename=None):

        super(MultiTaskLearner, self).fit(metatrain, metavalid, n_epochs, steps_per_epoch,
                                        max_episodes, batch_size, log_filename, checkpoint_filename)


if __name__ == '__main__':
    pass
