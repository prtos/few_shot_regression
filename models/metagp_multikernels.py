from .gp import *
from .utils import KL_div_diag_multivar_normals, sigm_heating, normal_entropy
from .metakrr_multikernels import MetaKrrMultiKernelsNetwork, MetaKrrMultiKernelsLearner

# debug in command-line with: import pdb; pdb.set_trace()


class MetaGPMultiKernelsNetwork(MetaKrrMultiKernelsNetwork):
    def __init__(self, *args, **kwargs):
        super(MetaGPMultiKernelsNetwork, self).__init__(*args, **kwargs)

    def __forward(self, episode):
        # training of the episode
        x_train, y_train = episode['Dtrain']
        task_descr = episode['task_descr']
        if self.meta_training:
            x_test, y_test = episode['Dtest']
            x_train, y_train = torch.cat([x_train, x_test]), torch.cat([y_train, y_test])
            self.conditionned_feat_extract.train()
            mu, log_var = self.conditionned_feat_extract.compute_task_mu_logvar(task_descr, x_train, y_train)
            if self.use_task_var:
                kl = KL_div_diag_multivar_normals(mu, log_var, self.prior_mu, self.prior_log_var)
                p = sigm_heating(self.step, self.beta_kl)
            else:
                kl, p = 0, 0
            self.conditionned_feat_extract.train()
            phis = self.conditionned_feat_extract(x_train, mu, log_var)
            learner = GPLearner(self.l2)
            learner.fit(phis, y_train)
            lml = learner.log_marginal_likelihood()
            self.reg_loss += lml / 2
            self.kl_loss += kl / 2
            return -learner.log_marginal_likelihood() - p*kl
        else:
            x_test, _ = episode['Dtest']
            n, batch_size = x_test.size(0), 16
            self.conditionned_feat_extract.train()
            mu, log_var = self.conditionned_feat_extract.compute_task_mu_logvar(task_descr, x_train, y_train)

            n_sampling = 10 if self.use_task_var else 1
            res = []
            for _ in range(n_sampling):
                # training of the episode
                self.conditionned_feat_extract.train()
                phis = self.conditionned_feat_extract(x_train, mu, log_var)
                learner = GPLearner(self.l2)
                learner.fit(phis, y_train)

                # Testing of the episode
                self.conditionned_feat_extract.eval()
                y_pred = torch.cat([learner(self.conditionned_feat_extract(x_test[i:i + batch_size], mu, log_var))
                                    for i in range(0, n, batch_size)])
                res.append(y_pred)
            y_pred = torch.mean(torch.stack(res), dim=0)
            return y_pred


class MetaGPMultiKernelsLearner(MetaKrrMultiKernelsLearner):

    def __init__(self, *args, **kwargs):
        super(MetaGPMultiKernelsLearner, self).__init__(*args, **kwargs)


if __name__ == '__main__':
    pass
