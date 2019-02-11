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
                 regularize_phi=False, fixe_hps=False, do_cv=False, device='cuda'):
        """
        In the constructor we instantiate an lstm module
        """
        super(MetaKrrSingleKernelNetwork, self).__init__()
        self.feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
        self.kernel = kernel
        self.do_cv = do_cv
        self.regularize_phi = regularize_phi
        self.device = device
        self.fixe_hps = fixe_hps

        if (not fixe_hps) and (not do_cv):
            self.l2 = Parameter(torch.FloatTensor([l2]).to(device))
        else:
            self.l2 = torch.FloatTensor([l2]).to(device)
            if do_cv and fixe_hps:
                self.l2s = torch.FloatTensor([l2]).to(device)
            
        if not do_cv:
            if fixe_hps:
                if kernel == 'rbf':
                    self.kernel_params = dict(gamma=torch.FloatTensor([gamma]).to(device))                  
                elif kernel == 'sm':
                    #todo: Need to finish this
                    self.kernel_params = dict(gamma=torch.FloatTensor([gamma]).to(device))
                else:
                    self.kernel_params = dict()
            else:
                if kernel == 'rbf':
                    self.kernel_params = ParameterDict(
                        dict(gamma=Parameter(torch.FloatTensor([gamma]).to(device))))                 
                elif kernel == 'sm':
                    #todo: Need to finish this
                    self.kernel_params = ParameterDict(
                        dict(gamma=Parameter(torch.FloatTensor([gamma]).to(device))))                 
                else:
                    self.kernel_params = dict()
        self.phis_norms = []

    def __forward(self, episode):
        # training part of the episode
        self.feature_extractor.train()
        x_train, y_train = episode['Dtrain']
        m = len(x_train)
        phis = self.feature_extractor(x_train)
        if self.do_cv:
            l2s = torch.logspace(-4, 1, 10).to(self.device) if not self.fixe_hps else self.l2s
            if self.kernel == 'linear':
                kernels_params = dict()
            elif self.kernel == 'rbf':
                kernels_params = dict(gamma = torch.logspace(-4, 1, 10).to(self.device))
            elif self.kernel == 'sm':
                raise NotImplementedError
            else:
                raise NotImplementedError
            learner = KrrLearnerCV(l2s, self.kernel, dual=False, **kernels_params)
        else:
            # l2 = torch.clamp(self.l2, min=1e-3)
            l2 = torch.FloatTensor([self.l2]).to(self.device)
            kp = {k: torch.clamp(self.kernel_params[k], min=1e-6) for k in self.kernel_params}
            learner = KrrLearner(l2, self.kernel, dual=False, **kp)

        learner.fit(phis, y_train)

        # Testing part of the episode
        self.feature_extractor.eval()
        x_test, _ = episode['Dtest']
        n = len(x_test)
        bsize = 10
        res = torch.cat([learner(self.feature_extractor(x_test[i:i+bsize])) for i in range(0, n, bsize)])

        self.phis_norms.append(torch.norm(phis, dim=1))
        self.l2_ = learner.l2
        self.kernel_params_ = learner.kernel_params
        return res

    def forward(self, episodes):
        self.phis_norms = []
        res = [self.__forward(episode) for episode in episodes]
        return res


class MetaKrrSingleKernelLearner(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = MetaKrrSingleKernelNetwork(*args, **kwargs, device=device)
        super(MetaKrrSingleKernelLearner, self).__init__(network, optimizer, lr, 
                                                        weight_decay)

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
        res.update(dict(l2=self.model.l2_))
        res.update({k: self.model.kernel_params_[k] for k in self.model.kernel_params_})
        return loss, res

if __name__ == '__main__':
    pass
