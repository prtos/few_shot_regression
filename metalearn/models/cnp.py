import torch
from torch.optim import Adam
from torch.nn.functional import mse_loss, softplus
from torch.nn import MSELoss, Parameter, ParameterDict, Sequential, ReLU
from torch.distributions.multivariate_normal import MultivariateNormal
from pytoune.framework import Model
from metalearn.feature_extraction import FeaturesExtractorFactory, FcFeaturesExtractor
from .base import MetaLearnerRegression, MetaNetwork
from .utils import reset_BN_stats, to_unit
from .set_encoding import RegDatasetEncoder
from .conditioning import ConditionerFactory


class CNPNetwork(MetaNetwork):

    def __init__(self, feature_extractor_params, target_dim=1,
                 encoder_hidden_sizes=[128], decoder_hidden_sizes=[128],):
        """
        In the constructor we instantiate an lstm module
        """
        super(CNPNetwork, self).__init__()
        self.feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
        phi_dim = self.feature_extractor.output_dim
        self.encoder = FcFeaturesExtractor(input_size=(phi_dim + target_dim),
                                           hidden_sizes=encoder_hidden_sizes)
        self.decoder = FcFeaturesExtractor(input_size=self.encoder.output_dim + phi_dim,
                                           hidden_sizes=decoder_hidden_sizes + [target_dim * 2])

    @property
    def return_var(self):
        return True

    def forward(self, episodes):
        xs_train, ys_train = zip(*[episode['Dtrain'] for episode in episodes])
        xs_test, _ = zip(*[episode['Dtest'] for episode in episodes])
        train_sizes = [xs.size(0) for xs in xs_train]
        test_sizes = [xs.size(0) for xs in xs_test]
        xs_train = torch.cat(xs_train, dim=0)
        ys_train = torch.cat(ys_train, dim=0)
        xs_test = torch.cat(xs_test, dim=0)

        self.feature_extractor.train()
        phis_train = self.feature_extractor(xs_train)
        phis_ys_train = torch.cat((phis_train, ys_train), dim=1)
        encoded_phis_ys = self.encoder(phis_ys_train).split(train_sizes)
        data_repr = torch.cat([chunk.mean(dim=0, keepdim=True).expand(test_sizes[i], -1)
                               for i, chunk in enumerate(encoded_phis_ys)],
                              dim=0)

        self.feature_extractor.eval()
        phis_test = self.feature_extractor(xs_test)
        phis_repr_test = torch.cat((phis_test, data_repr), dim=1)
        ys_pred = self.decoder(phis_repr_test)

        # Get the mean an the variance and bound the variance
        mu, log_sigma = ys_pred.chunk(2, dim=1)
        sigma = 0.1 + 0.9 * softplus(log_sigma)
        return list(zip(mu.split(test_sizes, dim=0), sigma.split(test_sizes, dim=0)))


def log_pdf(y, mu, std):
    try:
        cov = torch.diag(std.view(-1))
        n = cov.shape[0]
        log_p = MultivariateNormal(mu.view(-1), cov).log_prob(y.view(-1)) / n
    except:
        print('Error when computing log_pdf')
        print('y', y.view(-1))
        print('mu', mu.view(-1))
        print('std', std.view(-1))
        exit(1)
    return log_p


class CNPLearner(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, dataenc_beta=1.0, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = CNPNetwork(*args, **kwargs)
        super(CNPLearner, self).__init__(network, optimizer, lr, weight_decay)

    def _compute_aux_return_loss(self, y_preds, y_tests):
        res = dict()
        mse = torch.mean(torch.stack([mse_loss(mu, y_test)
                                      for (mu, sigma), y_test in zip(y_preds, y_tests)]))
        loss = -1 * torch.mean(torch.stack([log_pdf(y_test, mu, sigma)
                                            for (mu, sigma), y_test in zip(y_preds, y_tests)]))

        res.update(dict(mse=mse, marginal_likelihood=loss))
        return loss, res


if __name__ == '__main__':
    pass
