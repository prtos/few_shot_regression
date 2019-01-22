import torch
import math
from torch.nn.functional import mse_loss


def compute_kernel(x, y, kernel='linear', gamma=None, 
                    mixture_ws=None, means=None, gammas=None):
    if kernel.lower() == 'linear':
        K = torch.mm(x, y.t())
    elif kernel.lower() == 'rbf':
        assert gamma is not None
        x_i = x.unsqueeze(1)
        y_j = y.unsqueeze(0)
        xmy = ((x_i - y_j) ** 2).sum(2)
        K = torch.exp(-1* gamma * xmy)
    elif kernel.lower() == 'sm':
        assert mixture_ws is not None
        assert means is not None
        assert gammas is not None
        x_i = x.unsqueeze(1)
        y_j = y.unsqueeze(0)
        d = (x_i - y_j)
        d = d.unsqueeze(0).expand(means.size(0), *d.shape)
        mixture_ws = mixture_ws.reshape(*mixture_ws.shape, 1, 1)
        means = means.reshape(*means.shape, 1, 1, 1)
        gammas = gammas.reshape(*gammas.shape, 1, 1, 1)
        temp = torch.cos(2*math.pi*d*means) * torch.exp(-2*(math.pi**2)*(d** 2)*gammas)
        K = (mixture_ws * temp.sum(dim=3)).sum(dim=0)      
    else:
        raise Exception('Unhandled kernel name')
    return K

def compute_rbf_kernels(x, y, gamma):
    x_i = x.unsqueeze(1)
    y_j = y.unsqueeze(0)
    x_minus_y = ((x_i - y_j) ** 2).sum(2)
    x_minus_y = x_minus_y.unsqueeze(0).expand(gamma.shape[-1], *x_minus_y.shape)
    Ks = torch.exp(-gamma.reshape(gamma.shape[0], 1, 1) * x_minus_y)
    return Ks


class KrrLearner(torch.nn.Module):

    def __init__(self, l2, kernel='linear', dual=True, **kernel_params):
        super(KrrLearner, self).__init__()
        self.l2 = l2
        self.alpha = None
        self.phis_train = None
        self.dual = dual
        self.kernel = kernel
        self.kernel_params = kernel_params

    def fit(self, phis, y):
        batch_size_train = phis.size(0)
        K = compute_kernel(phis, phis, kernel=self.kernel, **self.kernel_params)
        I = torch.eye(batch_size_train, dtype=K.dtype)

        try:
            tmp = torch.inverse(K + self.l2 * I)
        except:
            tmp = torch.inverse(K + I)
            print("Inversion problem")
        self.alpha = torch.mm(tmp, y)
        self.phis_train = phis
        if not self.dual:
            self.w = torch.mm(self.phis_train.t(), self.alpha)
        return self

    def forward(self, phis):
        K = compute_kernel(phis, self.phis_train, kernel=self.kernel, **self.kernel_params)
        return torch.mm(K, self.alpha)


def cv_and_best_hp(X, y, kernel, l2s, **kernels_params):
    n, k = X.shape[0], l2s.shape[0]
    if kernel == 'linear':
        Ks = compute_kernel(X, X, kernel='linear').unsqueeze(0).expand(k, n ,n)
    else:
        Ks = compute_rbf_kernels(X, X, **kernels_params)
    Ks = torch.cat(torch.unbind(Ks, dim=0), dim=0)
    
    temp = torch.arange(0, k*n, n).view(-1, 1, 1)
    train_row_idx = (torch.eye(n, n)==0).nonzero()[:, 1].view(n, n-1)
    train_row_idx = train_row_idx.unsqueeze(dim=0).expand(k, *train_row_idx.shape)
    train_row_idx = (train_row_idx + temp).reshape(k*n, n-1)
    train_col_idx = train_row_idx.unsqueeze(1).expand(k*n, n-1, n-1).reshape(-1, n-1) % n

    test_row_idx = torch.arange(n).unsqueeze(dim=1)
    test_row_idx = test_row_idx.unsqueeze(dim=0).expand(k, *test_row_idx.shape)
    test_row_idx = (test_row_idx + temp).reshape(k*n, 1)
    test_col_idx = train_row_idx.unsqueeze(1).expand(*test_row_idx.shape, n-1).reshape(-1, n-1) % n

    Ks_train = torch.gather(Ks[train_row_idx.view(-1)], dim=1, index=train_col_idx).reshape(k*n, n-1, n-1)
    Ks_test = torch.gather(Ks[test_row_idx.view(-1)], dim=1, index=test_col_idx).reshape(k*n, 1, n-1)

    y_ = y.unsqueeze(dim=0).expand(k, *y.shape).reshape(k*n, 1)
    y_train, y_test = y_[train_row_idx], y_[test_row_idx]
    I = torch.eye(n - 1, device=Ks_train.device)
    I = I.unsqueeze(0).expand(k, *I.shape)
    I = I.unsqueeze(1).expand(k, n, n-1, n-1).reshape(-1, n-1, n-1)
    l2 = l2s.unsqueeze(1).expand(-1, n).reshape(-1).view(k*n, 1, 1)
    alphas, _ = torch.gesv(y_train, (Ks_train + l2*I))
    y_preds = torch.bmm(Ks_test, alphas)
    loss = mse_loss(y_preds.view(-1), y_test.view(-1), reduction='none')
    loss = loss.reshape(k, n).mean(dim=1)
    
    best_idx = torch.argmin(loss)
    scores = dict(l2=l2s, score=loss, **kernels_params)
    best_kernel_params = {key: values[best_idx] for key, values in kernels_params.items()}
    return l2s[best_idx], best_kernel_params, scores
    

def cv_and_best_hp_loopy(X, y, gammas, l2s):
    res = []
    for gamma, l2 in zip(gammas, l2s):
        n = X.shape[0]
        K = compute_kernel(X, X, kernel='rbf', gamma=gamma)
        train_row_idx = (torch.eye(n, n)==0).nonzero()[:, 1].view(n, n-1)
        train_col_idx = train_row_idx.unsqueeze(1).expand(*train_row_idx.shape, n-1).reshape(-1, n-1)
        test_row_idx = torch.arange(n).unsqueeze(dim=1)
        test_col_idx = train_row_idx.unsqueeze(1).expand(*test_row_idx.shape, n-1).reshape(-1, n-1)
        # print(train_col_idx)
        # print(train_col_idx.shape)
        # print(test_col_idx)
        # print(test_col_idx.shape)
        # exit()
        Ks_train = torch.gather(K[train_row_idx.view(-1)], dim=1, index=train_col_idx).reshape(n, n-1, n-1)
        Ks_test = torch.gather(K[test_row_idx.view(-1)], dim=1, index=test_col_idx).reshape(n, 1, n-1)
        # print(Ks_train.shape, Ks_test.shape)

        y_train, y_test = y[train_row_idx], y[test_row_idx]
        # print(y_train.shape, y_test.shape)
        I = torch.eye(n - 1, device=Ks_train.device)
        alphas, _ = torch.gesv(y_train, (Ks_train + l2*I))
        # print(alphas)
        y_preds = torch.bmm(Ks_test, alphas)
        loss = mse_loss(y_preds, y_test, reduction='elementwise_mean')
        # print(loss.shape)
        # print(gamma, l2, loss)
        # exit()
        res.append(loss)
    loss = torch.stack(res)
    best_idx = torch.argmin(loss)
    # print(l2s[best_idx], gammas[best_idx])
    # return l2s[best_idx], gammas[best_idx]
    return loss


def generate_grid(x, y):
   grid = torch.stack([x.repeat(y.size(0)), y.repeat(x.size(0),1).t().contiguous().view(-1)],1)
   return grid

class KrrLearnerCV(KrrLearner):
    
    def __init__(self, l2s, kernel='linear', dual=True, **kernels_params):
        super(KrrLearnerCV, self).__init__(0, kernel=kernel, dual=dual)
        assert l2s is not None
        if kernel == 'linear':
            self.l2s = l2s.reshape(-1)
            self.kernels_params = dict()
        elif kernel == 'rbf':
            temp = generate_grid(l2s, kernels_params['gamma'])
            self.l2s, gammas = temp.t()
            self.kernels_params = dict(gamma=gammas)
        elif kernel == 'sm':
            raise NotImplementedError
        else: 
            raise NotImplementedError

        self.scores = None

    def fit(self, phis, y):
        self.l2, self.kernel_params, self.scores = cv_and_best_hp(phis, y, self.kernel, self.l2s, **self.kernels_params)
        super(KrrLearnerCV, self).fit(phis, y)
        return self