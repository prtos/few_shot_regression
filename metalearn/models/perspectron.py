import torch
import numpy as np
from torch.nn import Linear, Sequential, Hardtanh, Tanh, ReLU, Sigmoid, Parameter
from torch.nn.functional import mse_loss, log_softmax, nll_loss
from torch.optim import Adam
from pytoune.framework import Model
from metalearn.models.attention import StandardSelfAttention
from tensorboardX import SummaryWriter
from pytoune.framework.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, \
    BestModelRestore, TensorBoardLogger
from metalearn.models.utils import to_unit
from metalearn.models.base import MetaLearnerRegression, FeaturesExtractorFactory


# debug in command-line with: import pdb; pdb.set_trace()
def normal_entropy(mu, logvar):
    return logvar.sum() + 0.5*logvar.size(0)*np.log(2*np.pi*np.e)


def compute_normal_agreement(mu_a, mu_b, mu_z, std_a, std_b, std_z):
    var_a = std_a ** 2
    var_b = std_b ** 2
    var_z = std_z ** 2
    deno = (var_a * var_z) + (var_b * var_z) - (var_a * var_b)
    num = (mu_b * var_a * var_z) + (mu_a * var_b * var_z) - (mu_z * var_a * var_b)
    mu = num / deno
    std = torch.sqrt((var_a * var_b * var_z) / deno)

    r_a = (mu_a ** 2) / var_a
    r_b = (mu_b ** 2) / var_b
    r_z = (mu_z ** 2) / var_z
    r = (mu ** 2) / (std ** 2)
    return torch.sum(torch.log((std * std_z) / (std_a * std_b)) - 0.5 * (r_a + r_b - r_z - r))


class DatasetFeatureExtractor(torch.nn.Module):
    def __init__(self, feature_extractor, latent_space_dim, input_dim=1, target_dim=1, pooling_mode='mean'):
        super(DatasetFeatureExtractor, self).__init__()
        self.feature_extractor = feature_extractor
        self.latent_space_dim = latent_space_dim
        if self.feature_extractor is None:
            agg_input_dim = input_dim + target_dim
        else:
            agg_input_dim = self.feature_extractor.output_dim + target_dim
        self.agg = Sequential(
            StandardSelfAttention(agg_input_dim, latent_space_dim, pooling_function=None),
            ReLU(),
            # StandardSelfAttention(latent_space_dim, latent_space_dim, pooling_function=None),
            StandardSelfAttention(latent_space_dim, latent_space_dim, pooling_function=pooling_mode),
            ReLU()
        )
        # self.lin1 = Linear(agg_input_dim, latent_space_dim)


    def extract_and_pool(self, inputs, targets):
        phis = inputs if self.feature_extractor is None else self.feature_extractor(inputs)
        x = torch.cat((phis, targets), dim=1).unsqueeze(0)
        res = self.agg(x).squeeze(0)
        return res

    def forward(self, batch_of_set_x_y):
        features = [self.extract_and_pool(x, y) for (x, y) in batch_of_set_x_y]
        return torch.stack(features, dim=0)


class PerspectronEncoderNetwork(torch.nn.Module):
    def __init__(self, feature_extractor_params, latent_space_dim, input_dim=None, target_dim=1, pooling_mode='mean',
                 is_latent_discrete=True, n_discrete_states=10):
        super(PerspectronEncoderNetwork, self).__init__()

        if isinstance(feature_extractor_params, torch.nn.Module):
            feature_extractor = feature_extractor_params
        elif feature_extractor_params is None:
            feature_extractor = None
        elif isinstance(feature_extractor, dict):
            feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
            input_size = feature_extractor.output_dim
        else:
            raise Exception('Wrong parameter type')

        assert pooling_mode in ['mean', 'max']
        ls_dim = latent_space_dim
        z_dim = (ls_dim * n_discrete_states) if is_latent_discrete else ls_dim
        self.data_feature_extractor = DatasetFeatureExtractor(feature_extractor=feature_extractor,
            latent_space_dim=ls_dim, input_dim=input_dim, target_dim=target_dim, pooling_mode=pooling_mode)
        self.latent_space_dim = ls_dim
        self.is_latent_discrete = is_latent_discrete
        self.n_discrete_states = n_discrete_states
        self.output_dim = z_dim

        if self.is_latent_discrete:
            self.logit_encoder = Linear(ls_dim, ls_dim*n_discrete_states)
            self.logit_prior = torch.ones((ls_dim, n_discrete_states))/n_discrete_states
        else:
            self.mu_encoder = Linear(ls_dim, ls_dim)
            self.std_encoder = Sequential(Linear(ls_dim, ls_dim), Sigmoid())
            self.mu_prior = Parameter(torch.zeros(ls_dim))
            self.std_prior = Parameter(torch.ones(ls_dim))

    def forward(self, batch_of_episodes):
        n = len(batch_of_episodes)
        temp = [ep['Dtrain'] for ep in batch_of_episodes] + [ep['Dtest'] for ep in batch_of_episodes]
        phis_data = self.data_feature_extractor(temp)
        if self.is_latent_discrete:
            log_probs = log_softmax(self.logit_encoder(phis_data).view(-1, self.latent_space_dim, self.n_discrete_states), dim=2)
            log_probs_a, log_probs_b = torch.split(log_probs, n, dim=0)
            return log_probs_a, log_probs_b
        else:
            mus = self.mu_encoder(phis_data)
            stds = self.std_encoder(phis_data)
            mus_a, mus_b = torch.split(mus, n)
            stds_a, stds_b = torch.split(stds, n)
            return (mus_a, stds_a), (mus_b, stds_b)

    def get_prior(self):
        if self.is_latent_discrete:
            return log_softmax(self.logit_prior, dim=1)
        else:
            return self.mu_prior, self.std_prior

    def compute_agreement_matrix(self, latent_params_a, latent_params_b):
        if self.is_latent_discrete:
            assert latent_params_a.size(0) == latent_params_b.size(0)
            n = latent_params_a.size(0)
        else:
            assert latent_params_a[0].size(0) == latent_params_b[0].size(0) and latent_params_a[0].size(0) == latent_params_a[1].size(0)
            n = latent_params_a[0].size(0)

        if self.is_latent_discrete:
            log_probs_a, log_probs_b = latent_params_a, latent_params_b
            log_probs_prior = log_softmax(self.logit_prior, dim=1)

            #loop version
            # agreement_matrix = torch.zeros((n, n))
            # if torch.cuda.is_available():
            #     agreement_matrix = agreement_matrix.cuda()
            # for i, lp_a in enumerate(log_probs_a):
            #     for j, lp_b in enumerate(log_probs_b):
            #         agreement_matrix[i, j] = torch.sum(torch.logsumexp(lp_a + lp_b - log_probs_prior,
            #                                                            dim=1, keepdim=False))
            # no loop version
            temp = log_probs_a[:, None] + log_probs_b[None, :] - log_probs_prior
            agreement_matrix = torch.sum(torch.logsumexp(temp, dim=3, keepdim=False), dim=2)
        else:
            agreement_matrix = torch.zeros((n, n), device=self.mu_prior.device)
            prior_stds = torch.zeros(n, n, self.latent_space_dim)
            # print(len(latent_params_a))
            # exit()
            (mus_a, stds_a), (mus_b, stds_b) = latent_params_a, latent_params_b
            for i in range(mus_a.size(0)):
                for j in range(mus_b.size(0)):
                    mu_a, std_a, mu_b, std_b = mus_a[i], stds_a[i], mus_b[j], stds_b[j]
                    temp = (std_a * std_b) / torch.sqrt((std_a ** 2) + (std_b ** 2))
                    mu_z, std_z = self.mu_prior, torch.max(self.std_prior, temp)
                    agreement_matrix[i, j] = compute_normal_agreement(mu_a, mu_b, mu_z,
                                                                      std_a, std_b, std_z)
                    prior_stds[i, j] = std_z
        return agreement_matrix


def hloss_from_agreement_matrix(agreement_matrix, return_left_right=False):
    assert len(agreement_matrix.size()) == 2
    assert agreement_matrix.size(0) == agreement_matrix.size(1)
    n = agreement_matrix.size(0)
    left = - torch.mean(torch.diag(agreement_matrix), dim=0, keepdim=False)
    nd_idx = (torch.eye(n) == 0).nonzero()
    right = torch.logsumexp(agreement_matrix[nd_idx[:, 0], nd_idx[:, 1]], dim=0, keepdim=False) - np.log(n*(n-1))
    if return_left_right:
        return left, right
    return left + right


def acc_from_agreement_matrix(agreement_matrix):
    classes = torch.arange(agreement_matrix.size(0), dtype=torch.long)
    if torch.cuda.is_available():
        classes = classes.cuda()
    probs = log_softmax(agreement_matrix, dim=1)
    return torch.mean((torch.argmax(probs, dim=1) == classes).float())


class PerspectronEncoderLearner(MetaLearnerRegression):
    def __init__(self, *args, network=None, optimizer='adam', lr=0.001, weight_decay=0.0, **kwargs):
        if network is None:
            network = PerspectronEncoderNetwork(*args, **kwargs)
        super(PerspectronEncoderLearner, self).__init__(network, optimizer, lr, weight_decay)

    def _compute_aux_return_loss(self, y_outs, y_true=None):
        latent_params_a, latent_params_b = y_outs
        m = self.model.compute_agreement_matrix(latent_params_a, latent_params_b)
        acc = acc_from_agreement_matrix(m)
        hloss_left, hloss_right = hloss_from_agreement_matrix(m, return_left_right=True)
        hloss = hloss_left + hloss_right

        res = dict(mutual_info=-hloss, accuracy=acc,
                    hloss_left=hloss_left, hloss_right=hloss_right,
                    hloss=hloss)
        return hloss, res


if __name__ == '__main__':
    pass
