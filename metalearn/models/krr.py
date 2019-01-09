import torch
from torch.nn.functional import mse_loss


def compute_kernel(x, y, kernel='linear', gamma=1):
    if kernel.lower() == 'linear':
        K = torch.mm(x, y.t())
    elif kernel.lower() == 'rbf':
        x_i = x.unsqueeze(1)
        y_j = y.unsqueeze(0)
        xmy = ((x_i - y_j) ** 2).sum(2)
        K = torch.exp(-gamma * xmy)
    else:
        raise Exception('Unhandled kernel name')
    return K

def compute_rbf_kernels(x, y, gammas):
    x_i = x.unsqueeze(1)
    y_j = y.unsqueeze(0)
    x_minus_y = ((x_i - y_j) ** 2).sum(2)
    x_minus_y = x_minus_y.unsqueeze(0).expand(gammas.shape[-1], *x_minus_y.shape)
    Ks = torch.exp(-gammas.reshape(gammas.shape[0], 1, 1) * x_minus_y)
    return Ks


class KrrLearner(torch.nn.Module):

    def __init__(self, l2, gamma=0, kernel='linear', dual=True):
        super(KrrLearner, self).__init__()
        self.l2 = l2
        self.alpha = None
        self.phis_train = None
        self.dual = dual
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, phis, y):
        batch_size_train = phis.size(0)
        K = compute_kernel(phis, phis, gamma=self.gamma, kernel=self.kernel)
        I = torch.eye(batch_size_train, dtype=K.dtype)

        try:
            tmp = torch.inverse(K + self.l2 * I)
        except:
            raise Exception("Inversion problem")
        self.alpha = torch.mm(tmp, y)
        self.phis_train = phis
        if not self.dual:
            self.w = torch.mm(self.phis_train.t(), self.alpha)
        return self

    def forward(self, phis):
        K = compute_kernel(phis, self.phis_train, gamma=self.gamma, kernel=self.kernel)
        return torch.mm(K, self.alpha)


def cv_and_best_hp(X, y, gammas, l2s):
    n, k = X.shape[0], gammas.shape[0]
    Ks = torch.cat(torch.unbind(compute_rbf_kernels(X, X, gammas), dim=0), dim=0)
    
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
    scores = dict(l2=l2s, gamma=gammas, score=loss)
    return gammas[best_idx], l2s[best_idx], scores
    

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


class KrrLearnerCV(KrrLearner):
    
    def __init__(self, l2s, gammas, dual=True):
        super(KrrLearnerCV, self).__init__(0, 0, kernel='rbf', dual=dual)
        self.l2s = l2s.unsqueeze(0).expand(gammas.shape[0], -1).reshape(-1)
        self.gammas = gammas.unsqueeze(1).expand(-1, l2s.shape[0]).reshape(-1)
        self.scores = None

    def fit(self, phis, y):
        self.l2, self.gamma, self.scores = cv_and_best_hp(phis, y, self.gammas, self.l2s)
        super(KrrLearnerCV, self).fit(phis, y)
        return self