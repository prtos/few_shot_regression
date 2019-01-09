import torch, os, json
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import log_softmax, nll_loss, sigmoid, mse_loss
from torch.nn import Linear, Sequential, ReLU, Module, Tanh
from torch.optim import Adam
from tensorboardX import SummaryWriter
from pytoune.framework import Model
from pytoune.framework.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, \
    BestModelRestore, TensorBoardLogger
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
from metalearn.models.perspectron import PerspectronEncoderNetwork, PerspectronEncoderLearner, hloss_from_agreement_matrix
from metalearn.models.utils import to_unit, sigm_heating
from metalearn.models.base import MetaLearnerRegression, FeaturesExtractorFactory, MetaNetwork


# debug in command-line with: import pdb; pdb.set_trace()
def reparameterize(mu, var, n_samples=1, identical_samples=False, is_training=False):
    if is_training and var is not None:
        std = torch.std(var)
        if identical_samples:
            eps = torch.randn(*mu.shape).expand(n_samples, *mu.shape)
        else:
            eps = torch.randn(n_samples, *mu.shape)
        if mu.is_cuda:
            eps = eps.cuda()
        return mu + (std * eps)
    else:
        return mu.expand(n_samples, *mu.shape)


def log_pdf(y, mu, std):
    cov = torch.diag(std.view(-1))
    return MultivariateNormal(mu.view(-1), cov).log_prob(y.view(-1))


def batch_kl(mus, stds, prior_mu=None, prior_std=None):
    assert mus is not None, "mu1 should not be None"
    assert len(mus.shape) == 2
    if prior_mu is None:
        prior_mu = torch.zeros_like(mus[0])
    if prior_std is None:
        prior_std = torch.ones_like(mus[0])
    if stds is None:
        stds = torch.ones_like(mus)

    kl = torch.stack([kl_divergence(MultivariateNormal(mu, torch.diag(std)),
                   MultivariateNormal(prior_mu, torch.diag(prior_std)))
     for mu, std in zip(mus, stds)]).sum()

    return kl


def std_activation(pre_std):
    # return (sigmoid(pre_std) * 0.1) + 1e-3
    return sigmoid(pre_std - 5)*0.05 + 0.001
    # return sigmoid(pre_std)


class ResidualBlock(Module):
    def __init__(self, input_dim, block_depth=2, scale_down_factor=2.):
        super(ResidualBlock, self).__init__()
        intern_dim = int(input_dim / scale_down_factor)
        layers = [Tanh()]
        in_dim, out_dim = input_dim, intern_dim
        for i in range(block_depth -1):
            layers.append(Linear(in_dim, out_dim))
            layers.append(Tanh())
            in_dim, out_dim = out_dim, intern_dim
        layers.append(Linear(in_dim, input_dim))
        self.net = Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)
    

class LateFusion(torch.nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size, nb_layers=1):
        super(LateFusion, self).__init__()
        if nb_layers <= 0:
            self.od = input_size1 + input_size2
            self.fusion_net = None
        else:
            hidden_sizes = [hidden_size]*nb_layers
            first_projection = Linear(input_size1 + input_size2, hidden_size)
            fusion_layers = [first_projection]
            for out_size in hidden_sizes:
                fusion_layers.append(ResidualBlock(out_size))
            self.fusion_net = Sequential(*fusion_layers)
            self.od = hidden_size

    def forward(self, inputs):
        phis, zs = inputs
        new_inputs = torch.cat((phis, zs), dim=1)
        res = new_inputs if self.fusion_net is None else self.fusion_net(new_inputs)
        return res

    @property
    def output_dim(self):
        return self.od


class DeepPriorNetwork(MetaNetwork):
    def __init__(self, input_dim=None, feature_extractor_params=None, 
                 task_encoder_params=None,
                 fusion_layer_size=100, fusion_nb_layer=0, 
                 beta_kl=1.0, task_repr_dim=None):
        super(DeepPriorNetwork, self).__init__()
        if feature_extractor_params is None:
            self.feature_extractor = None
        else:
            self.feature_extractor = FeaturesExtractorFactory()(**feature_extractor_params)
            input_size = self.feature_extractor.output_dim

        if task_encoder_params is None:
            self.task_repr_extractor = None
            assert task_repr_dim is not None, 'task_repr_dim cant be None if task_descr_extractor_params is None'
        else:
            self.task_repr_extractor = PerspectronEncoderNetwork(self.feature_extractor, 
                                        is_latent_discrete=False, **task_encoder_params)
            task_repr_dim = self.task_repr_extractor.output_dim

        fnet = LateFusion(input_size, task_repr_dim, fusion_layer_size, nb_layers=fusion_nb_layer)
        out_layer = Linear(fnet.output_dim, 2)
        self.fusion_net = Sequential(fnet, out_layer)
        self.beta_kl = beta_kl
        self.meta_training = True

    def _forward(self, episode, task_repr_params, z_sampling=False, z_identical_samples=False):
        x_test, _ = episode['Dtest']
        zs = reparameterize(*task_repr_params, n_samples=x_test.shape[0],
            identical_samples=z_identical_samples, is_training=(self.training or z_sampling))
        phis = x_test if self.feature_extractor is None else self.feature_extractor(x_test)
        outs = self.fusion_net((phis, zs))
        y_mean, y_std = torch.split(outs, 1, dim=1)
        y_std = std_activation(y_std)
        return y_mean, y_std

    def eval_pass(self, episode, n_rep=10):
        if self.task_repr_extractor:
            tasks_repr_params = self.task_repr_extractor([episode])[0]
        else:
            temp = torch.stack([episode['task_descr']])
            tasks_repr_params = (temp, torch.zeros_like(temp))
        task_repr_params = tasks_repr_params[0][0], tasks_repr_params[1][0]
        return [self._forward(episode, task_repr_params, z_sampling=True, z_identical_samples=True)
                for _ in range(n_rep)]

    def forward(self, episodes):
        if self.task_repr_extractor:
            tasks_repr_params = self.task_repr_extractor(episodes)
        else:
            temp = torch.stack([ep['task_descr'] for ep in episodes])
            tasks_repr_params = (temp, torch.zeros_like(temp)), (temp, torch.zeros_like(temp))
        tasks_repr_params_train, _ = tasks_repr_params

        # task
        # l_tests = [episode['Dtest'][0].shape[0] for episode in episodes]
        # x_tests = torch.cat([episode['Dtest'][0] for episode in episodes], dim=0)
        # zs = torch.torch.cat([reparameterize(*mu_std, n_samples=l_tests[i], is_training=self.training)
        #                       for i, mu_std in enumerate(zip(*tasks_repr_params_train))])
        # phis = x_tests if self.feature_extractor is None else self.feature_extractor(x_tests)

        # outs = self.fusion_net((phis, zs))
        # y_mean, y_std = torch.split(outs, 1, dim=1)
        # y_std = std_activation(y_std)
        # res = list(zip(torch.split(y_mean, l_tests, dim=0), torch.split(y_std, l_tests, dim=0)))

        res = [self._forward(episode, task_repr_params)
               for episode, task_repr_params in zip(episodes, zip(*tasks_repr_params_train))]
               
        return res, tasks_repr_params

    def meta_train(self):
        self.meta_training = True

    def meta_eval(self):
        self.meta_training = False


class DeepPriorLearner(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, cotraining=False, weight_decay=0.0, **kwargs):
        network = DeepPriorNetwork(*args, **kwargs)
        super(DeepPriorLearner, self).__init__(network, optimizer, lr, weight_decay)
        self.cotraining = cotraining

    def _compute_aux_return_loss(self, y_preds, y_tests):
        y_preds_per_task, tasks_repr_params = y_preds
        tasks_repr_params_train = tasks_repr_params[0]

        if self.model.task_repr_extractor:
            ag_matrix = self.model.task_repr_extractor.compute_agreement_matrix(*tasks_repr_params)
            prior = self.model.task_repr_extractor.get_prior()
            kl = batch_kl(*tasks_repr_params_train, *prior).mean(dim=-1).mean(dim=-1)
        else:
            ag_matrix = torch.eye(len(y_tests))
            prior = None
            kl = 0

        lml = torch.mean(torch.stack([log_pdf(y_test.view(-1), y_pred[0].view(-1), y_pred[1].view(-1))
                         for y_pred, y_test in zip(y_preds_per_task, y_tests)]))
        mse = torch.mean(torch.stack([mse_loss(y_pred[0].view(-1), y_test.view(-1))
                         for y_pred, y_test in zip(y_preds_per_task, y_tests)]))
        
        norm_std = torch.mean(torch.stack([torch.norm(std)**2 for (_, std) in y_preds_per_task]))

        hloss = hloss_from_agreement_matrix(ag_matrix)
        kl_weight = sigm_heating(self.train_step, self.model.beta_kl, 30000) if self.model.training else 0
        kl_weight = torch.Tensor([kl])[0]
        loss = -lml + kl_weight*kl      # + norm_std
        if self.cotraining:
            loss = loss + hloss

        scalars = dict(encoder_milbo=-1*hloss,
                        kl_value=kl,
                        neg_log_marginal_likelihood=-lml,
                        mse=mse,
                        kl_weight=kl_weight,
                        norm_std=norm_std)
                        
        return loss, scalars

    # def train_test_split(self, dataset, test_size):
    #     return dataset.train_test_split(test_size=test_size)

    # def fit(self, metatrain, *args, pretrain=False, valid_size=0.25, n_epochs=100, steps_per_epoch=100,
    #         batch_size=32, log_filename=None, checkpoint_filename=None, tboard_folder=None, **kwargs):
    #     fit_params = locals()
    #     meta_train, meta_valid = metatrain.train_test_split(test_size=valid_size)
    #     meta_valid.train()
    #     meta_train.train()
    #     print("Number of train steps:", len(meta_train))
    #     print("Number of valid steps:", len(meta_valid))

    #     callbacks = [# EarlyStopping(patience=10, verbose=False),)
    #                  ReduceLROnPlateau(patience=0, factor=1/2., min_lr=1e-6),
    #                  BestModelRestore()]
    #     if log_filename:
    #         callbacks += [CSVLogger(log_filename, batch_granularity=False, separator='\t')]
    #     if checkpoint_filename:
    #         callbacks += [ModelCheckpoint(checkpoint_filename, monitor='val_loss', save_best_only=True,
    #                                       temporary_filename=checkpoint_filename+'temp')]

    #     if tboard_folder is not None:
    #         print('here.....\n')
    #         self.writer = SummaryWriter(tboard_folder)
    #         # init_params = {k: v for k, v in self.init_params.items() if isinstance(v, (int, float, str, list, dict))}
    #         # fit_params = {k: v for k, v in fit_params.items() if isinstance(v, (int, float, str, list, dict))}
    #         # self.writer.add_text('deep_prior/model_params', json.dumps(init_params, indent=2), 0)
    #         # self.writer.add_text('deep_prior/fit_params', json.dumps(fit_params, indent=2), 0)

    #     if pretrain:
    #         p = PerspectronEncoderLearner(network=self.model.task_repr_extractor, lr=self.lr)
    #         p.fit(meta_train, valid_size=valid_size, n_epochs=n_epochs,
    #               steps_per_epoch=steps_per_epoch, tboard_folder=tboard_folder, early_stop=True)
    #         # if not self.cotraining:
    #         #     self.optimizer = Adam((p for name, p in self.model.named_parameters()
    #         #                            if 'task_repr_extractor' not in name), weight_decay=1e-6, lr=self.lr)
    #         # else:
    #         self.optimizer = Adam([
    #             dict(params=(p for name, p in self.model.named_parameters()
    #                          if 'task_repr_extractor' in name), lr=self.lr/10.0),
    #             dict(params=(p for name, p in self.model.named_parameters()
    #                          if 'task_repr_extractor' not in name), lr=self.lr)
    #         ], weight_decay=1e-6)

    #     self.fit_generator(meta_train, meta_valid,
    #                        epochs=n_epochs,
    #                        steps_per_epoch=steps_per_epoch,
    #                        validation_steps=None,
    #                        callbacks=callbacks,
    #                        verbose=True)
    #     self.is_fitted = True
    #     return self

    # def plot_harmonics(self, episode, step, tag=''):
    #     batch_preds_per_z = self.model.eval_pass(episode)
    #     plt.figure(dpi=60)
    #     x_train, y_train = episode['Dtrain']
    #     x_test, y_test = episode['Dtest']
    #     x_test = to_numpy_vec(x_test)
    #     x_order = np.argsort(x_test)
    #     x_test = x_test[x_order]
    #     y_test = to_numpy_vec(y_test)[x_order]
    #     plt.plot(to_numpy_vec(x_train), to_numpy_vec(y_train), 'ro')
    #     plt.plot(x_test, y_test, '-')
    #     for y_mean, y_var in batch_preds_per_z:
    #         y_pred = to_numpy_vec(y_mean)[x_order]
    #         plt.plot(x_test, y_pred, '-')
    #         if y_var is not None:
    #             y_var = to_numpy_vec(y_var)[x_order]
    #             plt.fill_between(x_test, y_pred - y_var, y_pred + y_var)

    #     self.writer.add_figure(tag=tag, figure=plt.gcf(), close=True, global_step=step)

    # def evaluate(self, metatest, metrics=[mse_loss], tboard_folder=None):
    #     assert len(metrics) >= 1, "There should be at least one valid metric in the list of metrics "
    #     if tboard_folder is not None:
    #         print('here.....\n')
    #         self.writer = SummaryWriter(tboard_folder)
    #     metrics_per_dataset = {metric.__name__: {} for metric in metrics}
    #     metrics_per_dataset["size"] = dict()
    #     it = 0
    #     for batch in metatest:
    #         episodes = batch[0]
    #         y_preds, _, _, _ = self.model(episodes)
    #         y_tests = batch[1]
    #         for episode, y_test, (y_pred_mean, y_pred_std) in zip(episodes, y_tests, y_preds):
    #             ep_idx = episode['idx']
    #             is_one_dim_input = (episode['Dtrain'][0].size(1) == 1)
    #             if is_one_dim_input and np.random.binomial(1, 0.1, 1)[0] == 1 and it <= 5000:
    #                 it += 1
    #                 self.plot_harmonics(episode, tag='test'+str(ep_idx), step=it)
    #             ep_name_is_new = (ep_idx not in metrics_per_dataset["size"])
    #             for metric in metrics:
    #                 m_value = to_unit(metric(y_pred_mean, y_test))
    #                 if ep_name_is_new:
    #                     metrics_per_dataset[metric.__name__][ep_idx] = [m_value]
    #                 else:
    #                     metrics_per_dataset[metric.__name__][ep_idx].append(m_value)
    #             metrics_per_dataset['size'][ep_idx] = y_test.size(0)

    #     return metrics_per_dataset

    # def load(self, checkpoint_filename):
    #     self.load_weights(checkpoint_filename)
    #     self.is_fitted = True


if __name__ == '__main__':
    pass
