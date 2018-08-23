import torch, math
import numpy as np

TEST = False
if TEST:
    from torch.nn.functional import mse_loss
    from sklearn.gaussian_process.gpr import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import DotProduct
    from sklearn.metrics import mean_squared_error


def compute_kernel(x, y):
    return torch.mm(x, y.t())
    # if kernel.lower() == 'linear':
    #     K = torch.mm(x, y.t())
    # elif kernel.lower() == 'rbf':
    #     x_i = x.unsqueeze(1)
    #     y_j = y.unsqueeze(0)
    #     xmy = ((x_i - y_j) ** 2).sum(2)
    #     K = torch.exp(-gamma_init * xmy)
    # else:
    #     raise Exception('Unhandled kernel name')
    # return K


def eye_like(tensor):
    return torch.eye(*tensor.size(), out=torch.empty_like(tensor))


class GPLearner(torch.nn.Module):
    def __init__(self, l2):
        super(GPLearner, self).__init__()
        self.l2 = l2
        self.alpha = None
        self.phis_train = None
        self.normalize_y = False
        if TEST:
            self.gp_test = GaussianProcessRegressor(alpha=l2.data.numpy(), kernel=DotProduct(sigma_0=0),
                                                    optimizer=None, normalize_y=self.normalize_y)

    def fit(self, phis, y):
        self.phis_train = phis
        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = torch.mean(y, dim=0)
            self.y_train = y - self._y_train_mean
        else:
            self._y_train_mean = 0
            self.y_train = y

        self.K_ = compute_kernel(phis, phis)
        I = eye_like(self.K_)
        I.requires_grad = False
        # self.L_ = torch.potrf(self.K_ + self.l2 * I, upper=False)
        # self.alpha = torch.potrs(y_, self.L_, upper=False)
        self.K_inv_ = torch.inverse(self.K_ + self.l2 * I)
        self.alpha = torch.mm(self.K_inv_, self.y_train)

        if TEST:
            self.gp_test.fit(phis.data.numpy(), y.data.numpy())
        return self

    def log_marginal_likelihood(self):
        # lml = -0.5 * torch.sum(self.alpha * self.y_train, dim=0)
        # lml -= torch.log(torch.diag(self.L_)).sum()
        # lml -= 0.5 * self.y_train.size(0) * np.log(2*np.pi)
        I = eye_like(self.K_)
        I.requires_grad = False
        lml = -0.5 * torch.sum(self.y_train * torch.mm(self.K_inv_, self.y_train))
        lml -= torch.log(torch.potrf(self.K_ + self.l2 * I, upper=False).diag()).sum()
        lml -= 0.5 * self.y_train.size(0) * np.log(2 * np.pi)
        if TEST:
            l = self.gp_test.log_marginal_likelihood()
            print(lml, l)
            assert self.gp_test.alpha == self.l2
            assert np.allclose(lml.sum().data.numpy(), l, atol=1e-5), \
                "Safety check failure because lml = {} but we expect {}".format(lml.sum().data.numpy(), l)
        return lml.sum()

    def forward(self, phis):
        K_trans = compute_kernel(phis, self.phis_train)
        y_mean = torch.mm(K_trans, self.alpha)
        y_mean = y_mean + self._y_train_mean

        # L_inv = torch.trtrs(self.L_, torch.eye(self.y_train.size(0)), upper=False)[0]
        # K_inv = torch.mm(L_inv, self.L_)
        y_var = torch.diag(compute_kernel(phis, phis))
        y_var -= torch.sum(torch.mm(K_trans, self.K_inv_) * K_trans, dim=1)
        if TEST:
            m, v = self.gp_test.predict(phis.data.numpy(), return_std=True)
            assert np.allclose(y_mean.data.numpy(), m), "Safety check failure because y_mean = {} but we expect {}".format(y_mean.data.numpy(), m)
            assert np.allclose(y_var.data.numpy(), v), "Safety check failure because y_var = {} but we expect {}".format(y_var.data.numpy(), v)

        return y_mean, y_var


if __name__ == '__main__':
    from torch.nn.functional import mse_loss
    from sklearn.gaussian_process.gpr import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import DotProduct
    from sklearn.metrics import mean_squared_error

    # Batch training test: Let's learn hyperparameters on a sine dataset, but test on a sine dataset and a cosine dataset
    # in parallel.
    train_x1 = torch.linspace(0, 1, 11).unsqueeze(-1)
    train_y1 = torch.sin(train_x1.data * (2 * math.pi))
    test_x1 = torch.linspace(0, 1, 51).unsqueeze(-1)
    test_y1 = torch.sin(test_x1.data * (2 * math.pi))
    print(train_x1.size(), train_y1.size(), test_x1.size(), test_y1.size())

    train_x2 = torch.linspace(0, 1, 11).unsqueeze(-1)
    train_y2 = torch.cos(train_x2.data * (2 * math.pi)).squeeze()
    test_x2 = torch.linspace(0, 1, 51).unsqueeze(-1)
    test_y2 = torch.cos(test_x2.data * (2 * math.pi)).squeeze()

    model = GPLearner(l2=1)
    model.fit(train_x1, train_y1)
    y_pred, y_var = model(test_x1)

    model_true = GaussianProcessRegressor(alpha=1, kernel=DotProduct(sigma_0=0), optimizer=None)
    model_true.fit(train_x1.data.numpy(), train_y1.data.numpy())
    y_pred_true = model_true.predict(test_x1.data.numpy())
    lml = model_true.log_marginal_likelihood()

    print(mse_loss(y_pred, test_y1).data.numpy())
    print(mean_squared_error(y_pred_true, test_y1.data.numpy()))
    assert np.allclose(mse_loss(y_pred, test_y1).data.numpy(), mean_squared_error(y_pred_true, test_y1.data.numpy()))
    print(lml)
    print(model.log_marginal_likelihood().data.numpy())