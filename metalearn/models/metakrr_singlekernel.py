import torch
from torch.nn import MSELoss, Parameter, ParameterDict
from torch.nn.functional import mse_loss
from torch.optim import Adam
from pytoune.framework import Model
from .base import MetaLearnerRegression, FeaturesExtractorFactory, MetaNetwork
from .krr import KrrLearner, KrrLearnerCV
from .utils import reset_BN_stats, to_unit



class MetaKrrSingleKernelNetwork(MetaNetwork):

    def __init__(self, feature_extractor_params, l2=0.1, gamma=0.1, kernel='linear',
                 regularize_phi=False, do_cv=False):
        """
        In the constructor we instantiate an lstm module
        """
        super(MetaKrrSingleKernelNetwork, self).__init__()
        self.feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
        self.kernel = kernel
        self.do_cv = do_cv
        self.regularize_phi = regularize_phi

        if not do_cv:
            module_device = next(self.feature_extractor.parameters()).device
            self.l2 = Parameter(torch.FloatTensor([l2], device=module_device))
            if kernel == 'rbf':
                self.kernel_params = ParameterDict(
                    dict(gamma=Parameter(torch.FloatTensor([gamma], device=module_device))))
            elif kernel == 'sm':
                #todo: Need to finish this
                self.kernel_params = ParameterDict(
                    dict(gamma=Parameter(torch.FloatTensor([gamma], device=module_device))))
            else:
                self.kernel_params = dict()
        self.phis_norms = []

    def __forward(self, episode):
        # training part of the episode
        self.feature_extractor.train()
        x_train, y_train = episode['Dtrain']
        phis = self.feature_extractor(x_train)
        if self.do_cv:
            l2s = torch.logspace(-4, 1, 10)
            kernels_params = dict()
            if self.kernel == 'rbf':
                kernels_params = dict(gamma = torch.logspace(-4, 1, 10))
            elif self.kernel == 'sm':
                raise NotImplementedError
            else:
                raise NotImplementedError
            learner = KrrLearnerCV(l2s, self.kernel, dual=False, **kernels_params)
        else:
            learner = KrrLearner(self.l2, self.kernel, dual=False, **self.kernel_params)
        learner.fit(phis, y_train)

        # Testing part of the episode
        self.feature_extractor.eval()
        x_test, _ = episode['Dtest']
        n = len(x_test)
        bsize = 64
        res = torch.cat([learner(self.feature_extractor(x_test[i:i+bsize])) for i in range(0, n, bsize)])

        self.phis_norms.append(torch.norm(phis, dim=1))
        return res

    def forward(self, episodes):
        self.phis_norms = []
        res = [self.__forward(episode) for episode in episodes]
        return res


class MetaKrrSingleKernelLearner(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, **kwargs):
        network = MetaKrrSingleKernelNetwork(*args, **kwargs)
        super(MetaKrrSingleKernelLearner, self).__init__(network, optimizer, lr, weight_decay)

    def _compute_aux_return_loss(self, y_preds, y_tests):
        res = dict()
        loss = torch.mean(torch.stack([mse_loss(y_pred, y_test) 
                    for y_pred, y_test in zip(y_preds, y_tests)]))

        res.update(dict(mse=loss))
        if self.model.training:
            x = torch.mean(torch.cat(self.model.phis_norms))
            res.update(dict(phis_norm=x))
            if self.model.regularize_phi:
                loss = loss + x

        return loss, res

if __name__ == '__main__':
    pass
