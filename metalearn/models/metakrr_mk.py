import torch
from torch.nn import MSELoss, Parameter, ParameterDict, Sequential, ReLU
from torch.nn.functional import mse_loss
from torch.optim import Adam
from pytoune.framework import Model
from .base import MetaLearnerRegression, FeaturesExtractorFactory, MetaNetwork
from .krr import KrrLearner, KrrLearnerCV
from .utils import reset_BN_stats, to_unit
from .set_encoding import RegDatasetEncoder
from .conditioning import ConditionerFactory


class MetaKrrMKNetwork(MetaNetwork):
    TRAIN = 0
    DESCR = 1
    BOTH = 2

    def __init__(self, feature_extractor_params, condition_on='train_samples',
                 task_descr_extractor_params=None,
                 dataset_encoder_params=None, l2=0.1,
                 regularize_dataencoder=False, hp_mode='fixe',
                 conditioner_mode='cat', conditioner_params=None, device='cuda'):
        """
        In the constructor we instantiate an lstm module
        """
        super(MetaKrrMKNetwork, self).__init__()
        if condition_on.lower() in ['train', 'train_samples']:
            assert dataset_encoder_params is not None, 'dataset_encoder_params must be specified'
            self.condition_on = self.TRAIN
        elif condition_on.lower() in ['descr', 'task_descr']:
            assert task_descr_extractor_params is not None, 'task_descr_extractor_params must be specified'
            self.condition_on = self.DESCR
        elif condition_on.lower() in ['both']:
            assert dataset_encoder_params is not None, 'dataset_encoder_params must be specified'
            assert task_descr_extractor_params is not None, 'task_descr_extractor_params must be specified'
            self.condition_on = self.BOTH
        else:
            raise ValueError('Invalid option for parameter condition_on')

        self.feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
        self.task_descr_extractor, self.dataset_encoder = None, None
        tde_dim, de_dim = None, None
        if self.condition_on in [self.DESCR, self.BOTH]:
            self.task_descr_extractor = FeaturesExtractorFactory()(**task_descr_extractor_params)
            tde_dim = self.task_descr_extractor.output_dim
        if self.condition_on in [self.TRAIN, self.BOTH]:
            self.dataset_encoder = RegDatasetEncoder(input_dim=self.feature_extractor.output_dim, **dataset_encoder_params)
            de_dim = self.dataset_encoder.output_dim
        cpa = dict() if conditioner_params is None else conditioner_params

        self.conditioner = ConditionerFactory()(conditioner_mode, self.feature_extractor.output_dim, tde_dim, de_dim, **cpa)
        self.kernel = 'linear'
        self.hp_mode = hp_mode
        self.device = device
        self.regularize_dataencoder = regularize_dataencoder

        if hp_mode.lower() in ['fixe', 'fixed', 'f']:
            self.hp_mode = 'f'
            self.l2 = torch.FloatTensor([l2]).to(device)
        elif hp_mode.lower() in ['learn', 'learned', 'l']:
            self.hp_mode = 'l'
            self.l2 = Parameter(torch.FloatTensor([l2]).to(device))
        else:
            raise Exception('hp_mode should be one of those: fixe, learn')

    def get_condition(self, episode, return_phi_train=False, use_test_partition=False):
        x_train, y_train = episode['Dtest'] if use_test_partition else episode['Dtrain']
        m = len(x_train)
        phis = self.feature_extractor(x_train)
        temp = []
        if self.dataset_encoder is not None:
            data_phi = self.dataset_encoder([(phis, y_train)])
            temp.append(data_phi)
        if self.task_descr_extractor is not None:
            if 'task_descr' not in episode:
                raise RuntimeError('The episode should have a task_descr attribute')
            task_descr = episode['task_descr']
            if task_descr is None:
                raise RuntimeError('The task descriptor is not speciified for an episode')
            task_phi = self.task_descr_extractor(task_descr.unsqueeze(0))
            temp.append(task_phi)
        if len(temp) == 2:
            condition = torch.cat(temp, dim=1)
        else:
            condition = temp[0]
        condition = condition.expand(phis.size(0), condition.size(1))
        if return_phi_train:
            return condition, phis
        return condition

    def __forward(self, episode):
        # training part of the episode
        self.feature_extractor.train()
        condition, phis = self.get_condition(episode, return_phi_train=True)
        task_descr = episode['task_descr']
        _, y_train = episode['Dtrain']
        phis = self.conditioner(phis, condition)

        l2 = torch.clamp(self.l2, min=1e-3)
        learner = KrrLearner(l2, self.kernel, dual=False, **kp)
        learner.fit(phis, y_train)

        # Testing part of the episode
        self.feature_extractor.eval()
        x_test, _ = episode['Dtest']
        n, bsize = len(x_test), 10
        res = torch.cat([learner(self.conditioner(self.feature_extractor(x_test[i:i + bsize]), condition))
                         for i in range(0, n, bsize)])
        self.l2_ = l2
        if self.regularize_dataencoder and self.training:
            condition_test = self.get_condition(episode, use_test_partition=True)
            self.train_test_regs.append(torch.norm(condition_test[0] - condition[0], keepdim=True))

        return res

    def forward(self, episodes):
        self.train_test_regs = []
        res = [self.__forward(episode) for episode in episodes]
        return res


class MetaKrrMKLearner(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, dataenc_beta=1.0, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = MetaKrrMKNetwork(*args, **kwargs, device=device)
        super(MetaKrrMKLearner, self).__init__(network, optimizer, lr,
                                               weight_decay)
        self.dataenc_beta = dataenc_beta

    def _compute_aux_return_loss(self, y_preds, y_tests):
        res = dict()
        loss = torch.mean(torch.stack([mse_loss(y_pred, y_test)
                                       for y_pred, y_test in zip(y_preds, y_tests)]))

        res.update(dict(mse=loss))
        if self.model.training:
            if self.model.regularize_dataencoder:
                x = torch.mean(torch.stack(self.model.train_test_regs))
                res.update(dict(dataencoder_reg=x))
                loss = loss + (self.dataenc_beta * x)

        res.update(dict(l2=self.model.l2_))
        return loss, res


if __name__ == '__main__':
    pass
