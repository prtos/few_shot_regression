import torch
from torch.nn import MSELoss, Parameter, ParameterDict
from torch.nn.functional import mse_loss
from torch.optim import Adam
from pytoune.framework import Model
from .base import MetaLearnerRegression, FeaturesExtractorFactory, MetaNetwork
from .krr import KrrLearner, KrrLearnerCV
from .utils import reset_BN_stats, to_unit


class MetaKrrSKNetwork(MetaNetwork):

    def __init__(self, feature_extractor_params, l2=0.1, gamma=0.1, kernel='linear',
                 hp_mode='fixe', device='cuda'):
        """
        In the constructor we instantiate an lstm module
        """
        super(MetaKrrSKNetwork, self).__init__()
        self.feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
        self.kernel = kernel
        self.hp_mode = hp_mode
        self.device = device

        if hp_mode.lower() in ['fixe', 'fixed', 'f']:
            self.hp_mode = 'f'
            self.l2 = torch.FloatTensor([l2]).to(device)
            self.kernel_params = dict()
            if kernel == 'rbf':
                self.kernel_params.update(dict(gamma=torch.FloatTensor([gamma]).to(device)))
            if kernel == 'sm':
                self.kernel_params.update(dict())  # todo: Need to finish this
        elif hp_mode.lower() in ['learn', 'learned', 'l']:
            self.hp_mode = 'l'
            self.l2 = Parameter(torch.FloatTensor([l2]).to(device))
            self.kernel_params = ParameterDict()
            if kernel == 'rbf':
                self.kernel_params.update(dict(gamma=Parameter(torch.FloatTensor([gamma]).to(device))))
            if kernel == 'sm':
                self.kernel_params.update(dict())  # todo: Need to finish this
        elif hp_mode.lower() in ['cv', 'valid', 'crossvalid']:
            self.hp_mode = 'cv'
            self.l2_grid = torch.logspace(-4, 1, 10).to(self.device) if not self.fixe_hps else self.l2s
            self.kernel_params_grid = dict()
            if self.kernel == 'rbf':
                self.kernel_params_grid.update(dict(gamma=torch.logspace(-4, 1, 10).to(self.device)))
            if self.kernel == 'sm':
                raise NotImplementedError
        else:
            raise Exception('hp_mode should be one of those: fixe, learn, cv')
        self.phis_norms = []

    def __forward(self, episode):
        # training part of the episode
        self.feature_extractor.train()
        x_train, y_train = episode['Dtrain']
        m = len(x_train)
        phis = self.feature_extractor(x_train)
        if self.hp_mode == 'cv':
            learner = KrrLearnerCV(self.l2_grid, self.kernel, dual=False, **self.kernel_params_grid)
        else:
            l2 = torch.clamp(self.l2, min=1e-3)
            kp = {k: torch.clamp(self.kernel_params[k], min=1e-6) for k in self.kernel_params}
            learner = KrrLearner(l2, self.kernel, dual=False, **kp)
        learner.fit(phis, y_train)

        # Testing part of the episode
        self.feature_extractor.eval()
        x_test, _ = episode['Dtest']
        n, bsize = len(x_test), 10
        res = torch.cat([learner(self.feature_extractor(x_test[i:i + bsize])) for i in range(0, n, bsize)])

        self.l2_ = learner.l2
        self.kernel_params_ = learner.kernel_params
        return res

    def forward(self, episodes):
        res = [self.__forward(episode) for episode in episodes]
        return res


class MetaKrrSKLearner(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = MetaKrrSKNetwork(*args, **kwargs, device=device)
        super(MetaKrrSKLearner, self).__init__(network, optimizer, lr,
                                               weight_decay)

    def _compute_aux_return_loss(self, y_preds, y_tests):
        res = dict()
        loss = torch.mean(torch.stack([mse_loss(y_pred, y_test)
                                       for y_pred, y_test in zip(y_preds, y_tests)]))

        res.update(dict(mse=loss))
        res.update(dict(l2=self.model.l2_))
        res.update({k: self.model.kernel_params_[k] for k in self.model.kernel_params_})
        return loss, res


if __name__ == '__main__':
    pass
