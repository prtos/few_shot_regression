import torch


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


class KrrLearner(torch.nn.Module):
    def __init__(self, l2):
        super(KrrLearner, self).__init__()
        self.l2 = l2
        self.alpha = None
        self.phis_train = None

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
        return self

    def forward(self, phis):
        K = compute_kernel(phis, self.phis_train)
        return torch.mm(K, self.alpha)