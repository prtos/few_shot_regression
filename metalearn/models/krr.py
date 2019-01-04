import torch


def compute_kernel(x, y, kernel='linear', gamma=1):
    return torch.mm(x, y.t())
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

    def __init__(self, l2, dual=True):
        super(KrrLearner, self).__init__()
        self.l2 = l2
        self.alpha = None
        self.phis_train = None
        self.dual = dual

    def fit(self, phis, y):
        batch_size_train = phis.size(0)
        K = compute_kernel(phis, phis)
        I = torch.eye(batch_size_train)
        if torch.cuda.is_available():
            I = I.cuda()

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
        K = compute_kernel(phis, self.phis_train)
        return torch.mm(K, self.alpha)


class KrrLearnerCV(torch.nn.Module):
    
    def __init__(self, l2s, gammas, dual=True):
        super(KrrLearnerCV, self).__init__()
        self.l2s = torch.cat([l2s for _ in gammas], dim=0)
        self.gammas = torch.cat([gammas for _ in l2s], dim=0)
        self.best_l2 = None
        self.best_gamma = None
        self.alpha = None
        self.phis_train = None
        self.dual = dual
        self.kernel_params = None

    def fit(self, phis, y):
        n, k = phis.shape[0], self.gammas.shape[0]
        temp = torch.arange(0, k*n, n).view(-1, 1, 1)
        train_idx = (torch.eye(n, n)==0).nonzero()[:, 1].view(n, n-1)
        train_idx = train_idx.unsqueeze(dim=0).expand(k, *train_idx.shape)
        train_idx = (train_idx + temp).reshape(k*n, n-1)
        test_idx = torch.arange(n).unsqueeze(dim=1)
        test_idx = test_idx.unsqueeze(dim=0).expand(k, *test_idx.shape)
        test_idx = (test_idx + temp).reshape(k*n, 1)
        # print(train_idx.shape)
        # print(test_idx.shape)
        phis_cv_train, phis_cv_test = phis[train_idx], phis[test_idx]
        y_cv_train, y_cv_test = y[train_idx], y[test_idx]

        batch_size_train = phis.size(0)
        Ks = compute_rbf_kernels(phis, phis, self.gammas)

        
        I = torch.eye(batch_size_train, device=Ks.device).unsqueeze(0).expand(*Ks.shape)

        alphas, _ = torch.gesv(y_cv_train, (batch_K + self.l2*I))

        self.alpha = torch.mm(tmp, y)
        self.phis_train = phis
        if not self.dual:
            self.w = torch.mm(self.phis_train.t(), self.alpha)
        return self

    def forward(self, phis):
        K = compute_kernel(phis, self.phis_train)
        return torch.mm(K, self.alpha)