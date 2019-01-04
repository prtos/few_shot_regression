import torch
from torch.nn import MSELoss
from torch.nn.functional import mse_loss
from torch.optim import Adam
from pytoune.framework import Model
from .base import MetaLearnerRegression, FeaturesExtractorFactory, MetaNetwork, to_unit
from .krr import KrrLearner
from .utils import reset_BN_stats



class MetaKrrSingleKernelNetwork(MetaNetwork):
    def __init__(self, feature_extractor_params, l2=0.1, regularize_w_pairs=False, 
                 regularize_phi=False, select_kernel=False):
        """
        In the constructor we instantiate an lstm module
        """
        super(MetaKrrSingleKernelNetwork, self).__init__()
        self.feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)

        self.l2 = torch.FloatTensor([l2])
        self.l2.requires_grad_(False)
        if torch.cuda.is_available():
            self.l2 = self.l2.cuda()

        self.regularize_w_pairs = regularize_w_pairs
        self.regularize_phi = regularize_phi
        self.phis_norms = []
        self.w_trn_tst_diff = []

    def __forward(self, episode):

        # training part of the episode
        self.feature_extractor.train()
        x_train, y_train = episode['Dtrain']
        phis = self.feature_extractor(x_train)
        learner = KrrLearner(self.l2, dual=False)
        learner.fit(phis, y_train)

        # Testing part of the episode
        self.feature_extractor.eval()
        x_test, _ = episode['Dtest']
        n = len(x_test)
        bsize = 64
        res = torch.cat([learner(self.feature_extractor(x_test[i:i+bsize])) for i in range(0, n, bsize)])

        if self.training:
            # training part of the episode
            self.feature_extractor.train()
            x_train, y_train = episode['Dtest']
            phis = self.feature_extractor(x_train)
            learner2 = KrrLearner(self.l2, dual=False)
            learner2.fit(phis, y_train)
            self.w_trn_tst_diff.append(torch.mean(torch.norm(learner.w - learner2.w, dim=0)))

        self.phis_norms.append(torch.norm(phis, dim=1))
        return res

    def forward(self, episodes):
        self.phis_norms = []
        self.w_trn_tst_diff = []
        res = [self.__forward(episode) for episode in episodes]
        return res


class MetaKrrSingleKernelLearner(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, **kwargs):
        network = MetaKrrSingleKernelNetwork(*args, **kwargs)
        super(MetaKrrSingleKernelLearner, self).__init__(network, optimizer, lr, weight_decay)

    def _compute_aux_return_loss(self, y_preds, y_tests):
        res = dict()
        if self.model.regularize_w_pairs and self.model.training:
            loss = torch.mean(torch.stack(y_preds))
        else:
            loss = torch.mean(torch.stack([mse_loss(y_pred, y_test) 
                    for y_pred, y_test in zip(y_preds, y_tests)]))

        res.update(dict(mse=loss))
        if self.model.training:
            x = torch.mean(torch.cat(self.model.phis_norms))
            res.update(dict(phis_norm=x))
            if self.model.regularize_phi:
                loss = loss + x

            x = torch.mean(torch.stack(self.model.w_trn_tst_diff))
            res.update(dict(diff_w_trn_tst=x))
            if self.model.regularize_w_pairs:
                loss = loss + x

        return loss, res

if __name__ == '__main__':
    pass
