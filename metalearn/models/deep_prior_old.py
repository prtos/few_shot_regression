from torch.nn.functional import log_softmax, nll_loss, sigmoid
from torch.nn import Linear
from torch.optim import Adam
from pytoune.framework import Model
from metalearn.feature_extraction import ClonableModule
from metalearn.models.base import *
from metalearn.models.krr import *
from metalearn.models.gp import *
from metalearn.models.conditioning import FeatureExtractorConditioner
from metalearn.models.utils import KL_div_diag_multivar_normals, sigm_heating, normal_entropy

# debug in command-line with: import pdb; pdb.set_trace()


def log_pdf(y, mu, var):
    lml = -torch.pow(y - mu, 2)/(2*var)
    lml += - 0.5 * torch.log(2*np.pi*var)
    # print(torch.sum(lml))
    # print(y, mu, var)
    return torch.sum(lml)


def var_activation(pre_var):
    return (sigmoid(pre_var) * 0.1) + 1e-3
    # return sigmoid(pre_var)


class DeepPriorNetwork(torch.nn.Module):
    def __init__(self, feature_extractor: ClonableModule, task_descr_extractor=None, conditioner_params=None,
                 use_task_var=False, use_data_encoder=True, beta_kl=1.0):
        super(DeepPriorNetwork, self).__init__()
        conditioner_params = {} if conditioner_params is None else conditioner_params
        self.feature_extractor = feature_extractor
        self.task_descr_extractor = task_descr_extractor
        self.step = 0
        self.writer = None
        self.beta_kl = beta_kl

        self.use_task_var = use_task_var
        self.use_data_encoder =use_data_encoder
        self.conditionned_feat_extract = FeatureExtractorConditioner(self.feature_extractor, self.task_descr_extractor,
                                                                     **conditioner_params, use_task_var=use_task_var,
                                                                     use_data_encoder=self.use_data_encoder)
        self.mean_layer = Linear(self.conditionned_feat_extract.output_dim, 1, bias=False)
        self.var_layer = Linear(self.conditionned_feat_extract.output_dim, 1, bias=False)
        if self.use_task_var:
            self.prior_mu = torch.zeros(self.conditionned_feat_extract.task_repr_size)
            self.prior_var = torch.ones(self.conditionned_feat_extract.task_repr_size)
            self.prior_mu.requires_grad_(False)
            self.prior_var.requires_grad_(False)
            if torch.cuda.is_available():
                self.prior_mu = self.prior_mu.cuda()
                self.prior_var = self.prior_var.cuda()

        self.meta_training = True

    def set_writer(self, writer):
        self.writer = writer

    def _forward(self, episode):
        task_descr = episode['task_descr']
        if self.meta_training:
            loss = 0
            for (x_train, y_train) in [episode['Dtrain'], episode['Dtest']]:
                mu, var = self.conditionned_feat_extract.compute_task_mu_var(task_descr, x_train, y_train)
                self.mu_tau.append(mu)
                if self.use_task_var:
                    kl = KL_div_diag_multivar_normals(mu, var, self.prior_mu, self.prior_var)
                    p = sigm_heating(self.step, self.beta_kl)
                else:
                    kl, p = 0, 0

                self.conditionned_feat_extract.eval()
                z = self.conditionned_feat_extract(x_train, mu, var)
                y_mean = self.mean_layer(z)
                # y_var = var_activation(self.var_layer(z))
                lml = log_pdf(y_train, y_mean, torch.Tensor([0.001]))
                loss += -lml + (p * kl)
                self.reg_loss += -lml/2
                self.kl_loss += kl/2
            return loss/2
        else:
            x_train, y_train = episode['Dtrain']
            x_test, _ = episode['Dtest']
            n, bs = x_test.size(0), 16
            mu, var = self.conditionned_feat_extract.compute_task_mu_var(task_descr, x_train, y_train)
            n_sampling = 10 if self.use_task_var else 1
            res = []
            for _ in range(n_sampling):
                self.conditionned_feat_extract.eval()
                zs = [self.conditionned_feat_extract(x_test[i:i + bs], mu, var, True) for i in range(0, n, bs)]
                y_mean = torch.cat([self.mean_layer(z) for z in zs], dim=0)
                y_var = torch.sqrt(var_activation(torch.cat([self.var_layer(z) for z in zs], dim=0)))
                y_pred_var = torch.cat((y_mean, y_var), dim=1)
                res.append(y_pred_var)
            #     print(y_pred_var)
            # exit()
            return res

    def forward(self, episodes):
        self.reg_loss, self.kl_loss = 0, 0
        self.mu_tau = []
        res = [self._forward(episode) for episode in episodes]
        N = len(res)
        if self.training:
            self.step += 1
            m = torch.cat(self.mu_tau)
            if self.writer is not None:
                scalars = dict(kl_beta=sigm_heating(self.step, self.beta_kl).data.cpu().numpy(),
                               regr_loss=self.reg_loss if isinstance(self.reg_loss, int) else (self.reg_loss/N).data.cpu().numpy(),
                               kl_loss=self.kl_loss if isinstance(self.kl_loss, float) else (self.kl_loss/N).data.cpu().numpy())
                for k, v in scalars.items():
                    self.writer.add_scalars('others/'+k, {k: v}, self.step)
                self.writer.add_scalars('others/' + 'tau', {'tau': m.mean(dim=0).mean().data.cpu().numpy()}, self.step)
                if self.step % 1000 == 0:
                    self.writer.add_embedding(m, global_step=self.step)
        return res

    def meta_train(self):
        self.meta_training = True

    def meta_eval(self):
        self.meta_training = False


class DeepPriorLearner(MetaLearnerRegression):

    def __init__(self, feature_extractor, task_descr_extractor=None, conditioner_params=None, use_task_var=False,
                 use_data_encoder=True, lr=0.001, beta_kl=1.0):
        super(DeepPriorLearner, self).__init__()

        self.network = DeepPriorNetwork(feature_extractor, task_descr_extractor, use_task_var=use_task_var,
                                        use_data_encoder=use_data_encoder,
                                        conditioner_params=conditioner_params, beta_kl=beta_kl)
        if torch.cuda.is_available():
            self.network.cuda()
        optimizer = Adam(
            [
                {'params': (p for name, p in self.network.named_parameters() if not name.endswith('reg_L2'))},
                {'params': (p for name, p in self.network.named_parameters() if name.endswith('reg_L2')),
                 'weight_decay': 1e-2}
            ],

            lr=lr)
        self.model = Model(self.network, optimizer, self.metaloss)

    def metaloss(self, y_preds, y_tests):
        if self.network.meta_training:
            res = torch.mean(torch.stack([loss for loss, y_test in zip(y_preds, y_tests)]))
        else:
            res = torch.mean(torch.stack([mse_loss(y_pred, y_test) for y_pred, y_test in zip(y_preds, y_tests)]))
        return res

    def fit(self, *args, **kwargs):
        self.network.meta_train()
        return super(DeepPriorLearner, self).fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        self.network.meta_eval()
        return super(DeepPriorLearner, self).evaluate(*args, **kwargs)


if __name__ == '__main__':
    pass
