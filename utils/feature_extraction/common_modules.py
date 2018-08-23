import torch
from torch.nn.functional import normalize
from torch.nn import Linear, Module, Sequential


class GaussianDropout(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.Tensor([alpha])

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1

            epsilon = torch.autograd.Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x


class ClonableModule(torch.nn.Module):
    def __init__(self):
        super(ClonableModule, self).__init__()

    def clone(self):
        raise NotImplementedError

    @property
    def output_dim(self):
        raise NotImplementedError


class UnitNormLayer(Module):
    def __init__(self):
        super(UnitNormLayer, self).__init__()

    def forward(self, x):
        return normalize(x)


class GlobalAvgPool1d(Module):
    def __init__(self):
        super(GlobalAvgPool1d, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=1)


class Transpose(Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)
        # return x.view(x.size(0), x.size(2), x.size(1))


class MyEmbedding(Module):
    # compared to torch.nn.Embedding this layer is twice differentiable
    def __init__(self, vocab_size, embedding_size):
        super(MyEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.net = Linear(vocab_size, embedding_size)

    def forward(self, x):
        x_new = x.view(-1, 1)
        x_one_hot = torch.zeros(x_new.size()[0], self.vocab_size).scatter_(1, x_new.data, 1.)
        y = self.net(torch.autograd.Variable(x_one_hot))
        # We have to reshape Y
        y = y.contiguous().view(*x.shape, y.size(-1))  # (samples, timesteps, output_size)
        return y


class EmbeddingViaLinear(Module):
    # compared to torch.nn.Embedding this layer is twice differentiable
    def __init__(self, vocab_size, embedding_size):
        super(EmbeddingViaLinear, self).__init__()
        self.vocab_size = vocab_size
        self.net = Linear(vocab_size, embedding_size)

    def forward(self, x):
        x_new = x.view(-1, x.size(2))
        y = self.net(x_new.float())
        # We have to reshape Y
        y = y.contiguous().view(*x.shape[:2], y.size(-1))  # (samples, timesteps, output_size)
        return y


class ResidualBlockMaker(Module):
    def __init__(self, base_module, downsample=None):
        super(ResidualBlockMaker, self).__init__()
        self.base_module = base_module
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.base_module(x)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out