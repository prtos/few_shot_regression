import unittest as ut
import torch
from torch.nn.functional import log_softmax, sigmoid
# from perspectron_eai.base.perspectron import compute_normal_agreement
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


class TestBatchOps(ut.TestCase):
    # def test_outer_product(self):
    #     a = torch.randn(3, 4)
    #     res = torch.einsum('ik,jk->ijk', (a, a))
    #
    #     expected = torch.stack([a[i] * a[j]for i in range(a.size(0)) for j in range(a.size(0))])
    #     expected = expected.view(3, 3, -1)
    #     print(1, expected.shape, res.shape)
    #
    #     self.assertEqual(res.shape, expected.shape)
    #     self.assertTrue(torch.allclose(res, expected))
    #
    # def test_outer_sum(self):
    #     a = torch.randn(3, 4)
    #     res = a[:, None] + a[None, :]
    #
    #     expected = torch.stack([a[i] + a[j] for i in range(a.size(0)) for j in range(a.size(0))])
    #     expected = expected.view(3, 3, -1)
    #     print(2, expected.shape, res.shape)
    #
    #     self.assertEqual(res.shape, expected.shape)
    #     self.assertTrue(torch.allclose(res, expected))

    def test_agreement_discrete_case(self):
        batch_size, n_latent_vars, n_latent_classes = 10, 4, 3
        log_probs_a = log_softmax(torch.randn(batch_size, n_latent_vars, n_latent_classes), dim=2)
        log_probs_b = log_softmax(torch.randn(batch_size, n_latent_vars, n_latent_classes), dim=2)
        log_probs_prior = log_softmax(torch.randn(n_latent_vars, n_latent_classes), dim=1)

        expected = torch.zeros((batch_size, batch_size))
        # loop version
        for i, lp_a in enumerate(log_probs_a):
            for j, lp_b in enumerate(log_probs_b):
                expected[i, j] = torch.sum(torch.logsumexp(lp_a + lp_b - log_probs_prior,
                                                                   dim=1, keepdim=False))

        # no-loop version
        temp = log_probs_a[:, None] + log_probs_b[None, :] - log_probs_prior
        res = torch.sum(torch.logsumexp(temp, dim=3, keepdim=False), dim=2)
        print(res.shape)
        self.assertEqual(res.shape, expected.shape)
        self.assertTrue(torch.allclose(res, expected))

    def test_agreement_normal_case(self):
        batch_size, latent_dim = 10, 4
        mus_a = torch.randn(batch_size, latent_dim)
        stds_a = sigmoid(torch.randn(batch_size, latent_dim))
        mus_b = torch.randn(batch_size, latent_dim)
        stds_b = sigmoid(torch.randn(batch_size, latent_dim))
        mu_prior = torch.randn(latent_dim)
        std_prior = sigmoid(torch.randn(latent_dim))


        expected = torch.zeros((batch_size, batch_size))
        expected_stds_z = torch.zeros((batch_size, batch_size, latent_dim))
        expected_mu = torch.zeros((batch_size, batch_size, latent_dim))
        expected_std = torch.zeros((batch_size, batch_size, latent_dim))
        for i in range(mus_a.size(0)):
            for j in range(mus_b.size(0)):
                mu_a, std_a, mu_b, std_b = mus_a[i], stds_a[i], mus_b[j], stds_b[j]
                temp = (std_a * std_b) / torch.sqrt((std_a ** 2) + (std_b ** 2))
                mu_z, std_z = mu_prior, torch.max(std_prior, temp)
                var_a = std_a ** 2
                var_b = std_b ** 2
                var_z = std_z ** 2
                deno = (var_a * var_z) + (var_b * var_z) - (var_a * var_b)
                num = (mu_b * var_a * var_z) + (mu_a * var_b * var_z) - (mu_z * var_a * var_b)

                mu = num / deno
                std = torch.sqrt((var_a * var_b * var_z) / deno)
                expected_mu[i, j] = mu
                expected_std[i, j] = std
                expected[i, j] = compute_normal_agreement(mu_a, mu_b, mu_z, std_a, std_b, std_z)
                expected_stds_z[i, j] = std_z
        print(torch.isnan(expected).sum())
        self.assertTrue(not torch.isnan(expected).any())

        vars_a = stds_a ** 2
        vars_b = stds_b ** 2
        temp = (stds_a[:, None] * stds_b[None, :]) / (torch.sqrt(vars_a[:, None] + vars_b[None, :]))
        stds_prior = torch.max(temp, std_prior)

        vars_prior = stds_prior ** 2
        v_a_b = (vars_a[:, None] * vars_b[None, :])
        deno = (vars_a[:, None] * vars_prior) + (vars_b[None, :] * vars_prior) - v_a_b
        num = ((vars_a[:, None] * mus_b[None, :]) * vars_prior) + \
              ((mus_a[:, None] * vars_b[None, :]) * vars_prior) - \
              (mu_prior * v_a_b)
        mus = num / deno
        stds = torch.sqrt((v_a_b * vars_prior) / deno)

        # print((v_a_b.shape, vars_prior))
        # # self.assertEqual(m.shape, expected_mu.shape)
        # self.assertTrue(torch.allclose(mus, expected_mu))
        # # self.assertEqual(stds.shape, expected_std.shape)
        # print(stds - expected_std)
        # self.assertTrue(torch.allclose(stds, expected_std))

        r_a = (mus_a ** 2) / vars_a
        r_b = (mus_b ** 2) / vars_b
        r_z = (mu_prior ** 2) / vars_prior
        r = (mus ** 2) / (stds ** 2)
        print(mu_prior.shape, r_a.shape, r_b.shape)
        print(stds.shape, stds_prior.shape)
        res = torch.sum(torch.log((stds * stds_prior) / (stds_a[:, None] * stds_b[None, :]))
                        - 0.5 * ((r_a[:, None] + r_b[None, :]) - r_z - r), dim=2)
        print(res.shape)
        self.assertEqual(res.shape, expected.shape)
        print(res - expected)
        self.assertTrue(torch.allclose(res, expected))