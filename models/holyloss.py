import torch
import numpy as np
from torch.nn import Linear, Sequential, Hardtanh, Tanh, ReLU
from torch.nn.functional import mse_loss, log_softmax, nll_loss
from torch.optim import Adam
from pytoune.framework import Model
from few_shot_regression.models.base import MetaLearnerRegression
from few_shot_regression.utils.feature_extraction import ClonableModule
from few_shot_regression.models.krr import *
from few_shot_regression.models.utils import KL_div_diag_multivar_normals
from few_shot_regression.models.attention import StandardSelfAttention

# debug in command-line with: import pdb; pdb.set_trace()


def normal_entropy(mu, logvar):
    return logvar.sum() + 0.5*logvar.size(0)*np.log(2*np.pi*np.e)


class HolyLossNetwork(torch.nn.Module):
    def __init__(self, feature_extractor: ClonableModule, mode='mean', use_net_for_mu=False):
        super(HolyLossNetwork, self).__init__()
        assert mode in ['mean_phi', 'mean_data_encoder', 'lin_reg', 'attention']
        self.feature_extractor = feature_extractor
        self.mode = mode
        self.use_net_for_mu = use_net_for_mu
        self.writer = None

        if mode == 'mean_phi':
            self.task_repr_size = 2*(self.feature_extractor.output_dim + 1)
        elif mode == 'mean_data_encoder':
            data_encoder_size = int(self.feature_extractor.output_dim / 2)
            self.data_encoder = Sequential(
                Linear(self.feature_extractor.output_dim+1, data_encoder_size),
                ReLU(),
                Linear(data_encoder_size, data_encoder_size))
            self.task_repr_size = data_encoder_size * 2
        elif mode == 'lin_reg':
            self.task_repr_size = feature_extractor.output_dim
        elif mode == 'attention':
            n = self.feature_extractor.output_dim
            self.agg = StandardSelfAttention(n+1, n, pooling_function='mean')
            self.task_repr_size = self.feature_extractor.output_dim

        if self.use_net_for_mu:
            self.task_encoder_mu_net = Sequential(Linear(self.task_repr_size, self.task_repr_size),
                                                  Hardtanh(min_val=-1, max_val=1))
        self.task_encoder_logvar_net = Sequential(Linear(self.task_repr_size, self.task_repr_size),
                                                  Hardtanh(min_val=-9, max_val=-2))
        self.step = 0

    def set_writer(self, writer):
        self.writer = writer

    def compute_task_mu_logvar(self, inputs, targets):
        phis = self.feature_extractor(inputs)
        if self.mode == 'mean_phi':
            x = torch.cat((targets, phis), dim=1)
            z = torch.cat((torch.mean(x, dim=0), torch.std(x, dim=0)), dim=0)
        elif self.mode == 'mean_data_encoder':
            x = torch.cat((targets, phis), dim=1)
            y = self.data_encoder(x)
            z = torch.cat((torch.mean(y, dim=0), torch.std(y, dim=0)), dim=0)
        elif self.mode == 'lin_reg':
            learner = KrrLearner(1/phis.size(0))
            learner.fit(phis, targets)
            z = (learner.alpha * phis).sum(dim=0)
        elif self.mode == 'attention':
            x = torch.cat((targets, phis), dim=1).unsqueeze(0)
            z = self.agg(x).squeeze(0)
        if self.use_net_for_mu:
            mu_task = self.task_encoder_mu_net(z)
        else:
            mu_task = z
        logvar_task = self.task_encoder_logvar_net(z)
        return mu_task, logvar_task

    def forward(self, episodes):
        self.step += 1
        Nep = len(episodes)
        acc_train, acc_test = [], []
        for i, episode in enumerate(episodes):
            acc_train.append(self.compute_task_mu_logvar(*episode['Dtrain']))
            acc_test.append(self.compute_task_mu_logvar(*episode['Dtest']))
        score_matrix = torch.zeros((Nep, Nep))
        classes = list(range(Nep))
        if torch.cuda.is_available():
            score_matrix = score_matrix.cuda()
        for i in range(Nep):
            for j in range(Nep):
                # score_matrix[i, j] = (acc_train[i][0] - acc_test[j][0]).pow(2).sum()
                score_matrix[i, j] = (KL_div_diag_multivar_normals(*acc_train[i], *acc_test[j]) +
                                      KL_div_diag_multivar_normals(*acc_test[j], *acc_train[i]) +
                                      normal_entropy(*acc_train[i]) +
                                      normal_entropy(*acc_test[j]))
        score_matrix = -1 * score_matrix
        classes = torch.Tensor(classes)
        if torch.cuda.is_available():
            classes = classes.cuda()
        classes = classes.to(torch.int64)
        probs = log_softmax(score_matrix, dim=1)
        if self.training and (self.writer is not None):
            enc_loss = nll_loss(probs, classes)
            acc = torch.mean((torch.argmax(probs, dim=1) == classes).float())
            scalars = dict(acc=acc.data.cpu().numpy(), loss=enc_loss.data.cpu().numpy())
            for k, v in scalars.items():
                self.writer.add_scalars('metrics/' + k, {k: v}, self.step)
            # self.writer.add_embedding(m, global_step=self.step)
        return probs, classes


class HolyLossLearner(MetaLearnerRegression):

    def __init__(self, *args, lr=0.01, **kwargs):
        super(HolyLossLearner, self).__init__()
        self.lr = lr
        self.network = HolyLossNetwork(*args, **kwargs)
        # for p in self.network.feature_extractor.parameters():
        #     p.requires_grad = False
        # optimizer = Adam((p for p in self.network.parameters() if p.requires_grad), lr=self.lr)
        if torch.cuda.is_available():
            self.network.cuda()
        optimizer = Adam(self.network.parameters(), lr=self.lr)
        self.model = Model(self.network, optimizer, self.metaloss, metrics=[acc])

    def metaloss(self, y_preds, y_tests):
        return nll_loss(*y_preds)


def acc(x,  y):
    probs, classes = x
    return torch.mean((torch.argmax(probs, dim=1) == classes).float())


if __name__ == '__main__':
    dataset = 'bdb'
    if dataset == 'uci':
        import few_shot_regression.configs.config_uci as cfg
    elif dataset == 'bdb':
        import few_shot_regression.configs.config_bdb as cfg
    from few_shot_regression.utils.datasets.loaders import load_episodic_bindingdb, load_episodic_uciselection, load_episodic_mhc
    from few_shot_regression.utils.feature_extraction import *
    loader = load_episodic_uciselection if dataset == 'uci' else load_episodic_bindingdb
    meta_train, meta_test = loader(max_examples_per_episode=10, batch_size=32)

    params = {'arch': 'fc', 'hidden_sizes': [32], 'input_size': 2, 'normalize_features': True}
    print(params)
    inner_class_dict = dict(
        tcnn=TcnnFeaturesExtractor,
        cnn=Cnn1dFeaturesExtractor,
        lstm=LstmFeaturesExtractor,
        fc=FcFeaturesExtractor,
        gcnn=GraphCnnFeaturesExtractor)
    fe_class = inner_class_dict[params['arch']]
    fe_params = {i: j for i, j in params.items() if i != 'arch'}
    feature_extractor = fe_class(**fe_params)
    model = HolyLossLearner(feature_extractor, mode='attention', use_net_for_mu=False)
    model.fit(meta_train, tboard_folder='tboard')

