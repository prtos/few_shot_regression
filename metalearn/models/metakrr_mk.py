import torch
from torch.nn import MSELoss, Parameter, ParameterDict, Sequential, ReLU
from torch.nn.functional import mse_loss
from torch.optim import Adam
from pytoune.framework import Model
from .base import MetaLearnerRegression, FeaturesExtractorFactory, MetaNetwork
from .krr import KrrLearner, KrrLearnerCV
from .utils import reset_BN_stats, to_unit
from .set_encoding import RegDatasetEncoder, AttentionLayer
from .conditioning import ConditionerFactory


class MetaKrrMKNetwork(MetaNetwork):
    TRAIN = 0
    DESCR = 1
    BOTH = 2

    def __init__(self, input_features_extractor_params,
                 target_features_extractor_params,
                 condition_on='train_samples',
                 task_descr_extractor_params=None, dataset_encoder_params=None,
                 hp_mode='fixe', l2=0.1,
                 conditioner_mode='cat', conditioner_params=None,
                 task_memory_size=None, softmax_coef=1, use_improvement_loss=True, device='cuda'):
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

        self.input_features_extractor = FeaturesExtractorFactory()(**input_features_extractor_params)
        self.target_features_extractor = FeaturesExtractorFactory()(**target_features_extractor_params)

        self.task_descr_extractor, self.dataset_encoder = None, None
        tde_dim, de_dim = None, None
        if self.condition_on in [self.DESCR, self.BOTH]:
            self.task_descr_extractor = FeaturesExtractorFactory()(**task_descr_extractor_params)
            tde_dim = self.task_descr_extractor.output_dim
        if self.condition_on in [self.TRAIN, self.BOTH]:
            self.dataset_encoder = RegDatasetEncoder(input_dim=self.input_features_extractor.output_dim, target_dim=self.target_features_extractor.output_dim, **dataset_encoder_params)
            de_dim = self.dataset_encoder.output_dim
        cpa = dict() if conditioner_params is None else conditioner_params

        self.conditioner = ConditionerFactory()(conditioner_mode, self.input_features_extractor.output_dim, tde_dim, de_dim, **cpa)
        self.kernel = 'linear'
        self.hp_mode = hp_mode
        self.device = device
        self.use_memory = (task_memory_size is not None)
        self.use_improvement_loss = use_improvement_loss

        if hp_mode.lower() in ['fixe', 'fixed', 'f']:
            self.hp_mode = 'f'
            self.l2 = torch.FloatTensor([l2]).to(device)
        elif hp_mode.lower() in ['learn', 'learned', 'l']:
            self.hp_mode = 'l'
            self.l2 = Parameter(torch.FloatTensor([l2]).to(device))
        else:
            raise Exception('hp_mode should be one of those: fixe, learn')

        if self.use_memory:
            condition_size = 0
            if tde_dim is not None:
                condition_size += tde_dim
            if de_dim is not None:
                condition_size += de_dim
            self.memory = Parameter(torch.FloatTensor(task_memory_size, condition_size)).to(device)
            torch.nn.init.xavier_uniform(self.memory)
            self.mem_controller = AttentionLayer(input_size=condition_size, key_size=condition_size, value_size=condition_size, pooling_function='mean', softmax_coef=softmax_coef)

    def get_condition(self, episode, return_phi_train=False, use_test_partition=False):
        x_train, y_train = episode['Dtest'] if use_test_partition else episode['Dtrain']
        m = len(x_train)
        # print('nb_examples', return_phi_train, m)
        phis = self.input_features_extractor(x_train)
        yy = self.target_features_extractor(y_train)
        temp = []
        if self.dataset_encoder is not None:
            data_phi = self.dataset_encoder([(phis, yy)])
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

        if self.use_memory:
            mem = self.memory.unsqueeze(0)
            condition, attention = self.mem_controller(condition.unsqueeze(0), mem, mem, return_attention=True)
            self.attention_weights.append(attention[0])

        # condition = condition.expand(phis.size(0), condition.size(1))

        if return_phi_train:
            return condition, phis

        return condition

    def __forward(self, episode):
        # training part of the episode
        self.input_features_extractor.train()
        condition, phis = self.get_condition(episode, return_phi_train=True)
        _, y_train = episode['Dtrain']
        n_train = len(y_train)
        phis_c = self.conditioner(phis, condition.expand(n_train, -1))
        phis_train = torch.stack([phis, phis_c])
        y_train = torch.stack([y_train, y_train])
        batch_K = torch.bmm(phis_train, phis_train.transpose(1, 2))
        Identity = torch.eye(n_train, device=batch_K.device)
        alphas, _ = torch.gesv(y_train, (batch_K + self.l2 * Identity))

        self.input_features_extractor.eval()
        x_test, y_test = episode['Dtest']
        n_test = len(y_test)
        phis = self.input_features_extractor(x_test)
        phis_c = self.conditioner(phis, condition.expand(n_test, -1))
        phis_test = torch.stack([phis, phis_c])
        batch_K = torch.bmm(phis_test, phis_train.transpose(1, 2))
        preds = torch.bmm(batch_K, alphas)
        res, res_c = torch.bmm(batch_K, alphas)

        mse = mse_loss(res, y_test)
        mse_c = mse_loss(res_c, y_test)
        self.losses.append((mse, mse_c, (mse_c / (mse + torch.tensor([1e-3])))))
        return res_c

    def forward(self, episodes):
        self.losses = []
        self.attention_weights = []
        res = [self.__forward(episode) for episode in episodes]
        return res


class MetaKrrMKLearner(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0,
                 **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = MetaKrrMKNetwork(*args, **kwargs, device=device)
        super(MetaKrrMKLearner, self).__init__(network, optimizer, lr,
                                               weight_decay)

    def _compute_aux_return_loss(self, y_preds, y_tests):
        res = dict()
        n = torch.FloatTensor([len(self.model.losses)])
        for i, (woc, wc, imp) in enumerate(self.model.losses):
            if i == 0:
                mse_wo_cond, mse_w_cond, improvement = woc, wc, imp
            else:
                mse_wo_cond = mse_wo_cond + woc
                mse_w_cond = mse_w_cond + wc
                improvement = improvement + imp

        if self.model.use_improvement_loss:
            loss = mse_wo_cond + mse_w_cond + improvement
            loss = loss / n
        else:
            loss = mse_w_cond / n

        mse_wo_cond = mse_wo_cond / n
        mse_w_cond = mse_w_cond / n
        improvement = improvement / n

        res.update(dict(mse_wo_cond=mse_wo_cond, mse_w_cond=mse_w_cond, improvement=improvement))
        # print(res, loss)
        res.update(dict(l2=self.model.l2))

        if len(self.model.attention_weights) > 0 and self.model.training:
            self.writer.add_embedding(torch.cat(self.model.attention_weights), global_step=self.train_step)

        return loss, res


if __name__ == '__main__':
    pass
