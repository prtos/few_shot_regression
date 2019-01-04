import torch
from torch.nn import Parameter
from torch.optim import Adam
from pytoune.framework import Model
from .base import MetaLearnerRegression, FeaturesExtractorFactory, MetaNetwork
from .gp import *
from .utils import reset_BN_stats


def activate_l2(l2):
    return torch.sigmoid(l2) + 1e-4


class MetaGPSingleKernelNetwork(MetaNetwork):
    def __init__(self, feature_extractor_params, l2=0.1):
        super(MetaGPSingleKernelNetwork, self).__init__()
        self.feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
        l2 = 0.1 if l2 is None else l2
        # self.pre_l2 = torch.FloatTensor([l2])
        # self.pre_l2.requires_grad_(True)
        self.pre_l2 = Parameter(torch.FloatTensor([l2]))
        if torch.cuda.is_available():
            self.pre_l2 = self.pre_l2.cuda()
        self.register_parameter('pre_l2', self.pre_l2)
        self.meta_training = False
        self.writer = None
        self.step = 0

    def set_writer(self, writer):
        self.writer = writer

    def __forward(self, episode):
        # training part of the episode
        self.l2 = activate_l2(self.pre_l2)
        reset_BN_stats(self.feature_extractor)
        if self.meta_training:
            loss = 0
            for (x_train, y_train) in [episode['Dtrain'], episode['Dtest']]:
                self.feature_extractor.train()
                phis = self.feature_extractor(x_train)
                learner = GPLearner(self.l2)
                learner.fit(phis, y_train)
                lml = learner.log_marginal_likelihood()
                loss += -lml / 2
            return loss / 2
        else:
            x_train, y_train = episode['Dtrain']
            self.feature_extractor.train()
            phis = self.feature_extractor(x_train)
            learner = GPLearner(self.l2)
            learner.fit(phis, y_train)
            # Testing part of the episode
            self.feature_extractor.eval()
            x_test, _ = episode['Dtest']
            n = len(x_test)
            batch_size = 64
            y_pred_and_var = torch.cat([learner(self.feature_extractor(x_test[i:i + batch_size]))
                                        for i in range(0, n, batch_size)], dim=0)
            # end of testing
            self.feature_extractor.train()
            return y_pred_and_var

    def forward(self, episodes):
        res = [self.__forward(episode) for episode in episodes]
        if self.training:
            self.step += 1
            if self.writer is not None:
                scalars = dict(l2_gp=self.l2.data.cpu().numpy(),
                               regr_loss=torch.mean(torch.stack(res)).data.cpu().numpy(),
                               )
                for k, v in scalars.items():
                    self.writer.add_scalars('others/'+k, {k: v}, self.step)
        return res

    def meta_train(self):
        self.meta_training = True

    def meta_eval(self):
        self.meta_training = False


class MetaGPSingleKernelLearner(MetaLearnerRegression):

    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, **kwargs):
        network = MetaGPSingleKernelNetwork(*args, **kwargs)
        super(MetaGPSingleKernelLearner, self).__init__(network, optimizer, lr, weight_decay)

    def _compute_loss_and_metrics(self, y_preds, y_tests):
        if self.network.meta_training:
            res = torch.mean(torch.stack([loss for loss, _ in zip(y_preds, y_tests)]))
        else:
            res = torch.mean(torch.stack([mse_loss(y_pred[:, 0], y_test) for y_pred, y_test in zip(y_preds, y_tests)]))
        return res

    def fit(self, *args, **kwargs):
        self.network.meta_train()
        return super(MetaGPSingleKernelLearner, self).fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        self.network.meta_eval()
        return super(MetaGPSingleKernelLearner, self).evaluate(*args, **kwargs)


if __name__ == '__main__':
    pass
