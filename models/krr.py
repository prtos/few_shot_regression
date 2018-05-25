import torch
from torch.autograd import Variable


def compute_kernel(x, y, kernel, gamma_init=1):
    if kernel.lower() == 'linear':
        K = torch.mm(x, y.t())
    elif kernel.lower() == 'rbf':
        x_i = x.unsqueeze(1)
        y_j = y.unsqueeze(0)
        xmy = ((x_i - y_j) ** 2).sum(2)
        K = torch.exp(-gamma_init * xmy)
    else:
        raise Exception('Unhandled kernel name')
    return K


class KrrLearner(torch.nn.Module):
    def __init__(self, l2_penalty, kernel='linear', center_kernel=False, gamma=1):
        super(KrrLearner, self).__init__()
        self.l2_penalty = l2_penalty
        self.alpha = None
        self.phis_train = None
        self.kernel = kernel
        self.center_kernel = center_kernel
        self.gamma = gamma

    def fit(self, phis, y):
        batch_size_train = phis.size(0)
        K = compute_kernel(phis, phis, self.kernel, self.gamma)
        I = torch.eye(batch_size_train)
        if torch.cuda.is_available():
            I = I.cuda()
        I = Variable(I)
        if self.center_kernel:
            self.H = I - (1/batch_size_train)
            K = torch.mm(torch.mm(self.H, K), self.H)
            self.y_mean = torch.mean(y)
            self.K = K
        else:
            self.y_mean = 0

        tmp = torch.inverse(K + self.l2_penalty * I)
        self.alpha = torch.mm(tmp, (y-self.y_mean))
        self.phis_train = phis
        self.w_norm = torch.norm(torch.mm(self.alpha.t(), self.phis_train), p=2)

    def forward(self, phis):
        K = compute_kernel(phis, self.phis_train, self.kernel)
        if self.center_kernel:
            K_mean = torch.mean(self.K, dim=1)
            K = torch.mm(K - K_mean, self.H)
            y = torch.mm(K, self.alpha) + self.y_mean
        else:
            y = torch.mm(K, self.alpha)
        return y
