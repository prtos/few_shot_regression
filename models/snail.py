import torch, numpy
from torch.nn.functional import mse_loss, tanh, sigmoid, softmax
from torch.nn import Conv1d, Sequential, Linear
from torch.optim import Adam
from pytoune.framework import Model
from .base import MetaLearnerRegression
from few_shot_regression.utils.feature_extraction.common_modules import ClonableModule, Transpose


class DenseBlock(torch.nn.Module):
    def __init__(self, in_channel, filters, dilatation_rate, kernel_size):
        super(DenseBlock, self).__init__()
        self.filters = filters
        self.dilatation_rate = dilatation_rate
        self.kernel_size = kernel_size

        pad = (self.dilatation_rate * (self.kernel_size - 1) + 1) // 2
        self.g = Conv1d(in_channel, filters, padding=pad,
                             kernel_size=kernel_size, dilation=self.dilatation_rate)
        self.f = Conv1d(in_channel, filters, padding=pad,
                             kernel_size=kernel_size, dilation=self.dilatation_rate)

    def forward(self, inputs):
        xg, xf = self.g(inputs), self.f(inputs)
        activations = tanh(xf) * sigmoid(xg)
        return torch.cat([inputs, activations], dim=1)

    @property
    def output_dim(self):
        return self.in_channel + self.kernel_size


class TCBlock(torch.nn.Module):
    def __init__(self, in_channel, L, filters):
        super(TCBlock, self).__init__()
        self.filters = filters
        layers = []
        layers.append(Transpose(1, 2))
        for i in range(1, int(numpy.ceil(numpy.log2(L)))):
            layers.append(DenseBlock(in_channel, filters, 2**i, 2))
            in_channel += filters
        layers.append(Transpose(1, 2))
        self.output_dim = in_channel
        self.net = Sequential(*layers)

    def forward(self, inputs):
        return self.net(inputs)


class AttentionBlock(torch.nn.Module):
    def __init__(self, input_size, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.key_size = key_size
        self.linear_keys = Linear(input_size, key_size)
        self.linear_query = Linear(input_size, key_size)
        self.linear_values = Linear(input_size, value_size)

    def forward(self, inputs):
        # want to make it more batch efficient
        # look at https://github.com/eladhoffer/seq2seq.pytorch/blob/master/seq2seq/models/modules/attention.py
        if not inputs.is_contiguous():
            inputs = inputs.contiguous()
        b_size, ts, dim = inputs.size()
        inputs_ = inputs.view(-1, dim)
        keys, query, values = self.linear_keys(inputs_), self.linear_query(inputs_), self.linear_values(inputs_)
        keys, query, values = keys.view(b_size, ts, -1), query.view(b_size, ts, -1), values.view(b_size, ts, -1)

        logits = torch.bmm(query, keys.transpose(1, 2))
        probs = logits / (self.key_size**0.5)

        causal_mask = torch.ones_like(probs.data[0]).byte().triu_(1).expand(b_size, -1, -1)
        causal_mask.requires_grad_(False)
        if torch.cuda.is_available():
            causal_mask = causal_mask.cuda()
        probs.masked_fill_(causal_mask, -1e9)
        probs = softmax(probs, dim=2)

        read = torch.bmm(probs, values)
        return torch.cat([inputs, read], dim=2)


class SNAILNetwork(torch.nn.Module):
    def __init__(self, input_transformer: ClonableModule, k, arch):
        """
        In the constructor we instantiate the snail module
        """
        super(SNAILNetwork, self).__init__()
        self.input_transformer = input_transformer
        self.arch = arch
        input_dim = self.input_transformer.output_dim + 1

        layers = []
        for i, (layer_type, layer_infos) in enumerate(arch):
            if layer_type == 'att':
                key_size, value_size = layer_infos
                layer = AttentionBlock(input_dim, key_size, value_size)
                input_dim = input_dim + value_size
            elif layer_type == 'tc':
                kernel_size = layer_infos
                layer = TCBlock(input_dim, k+1, kernel_size)
                input_dim = layer.output_dim
            else:
                raise Exception("Impossible to create that type of layer in this model")
            layers.append(layer)
        layers.append(Transpose(1, 2))
        layers.append(Conv1d(input_dim, 1, 1))
        layers.append(Transpose(1, 2))
        self.net = Sequential(*layers)

    def __forward(self, episode):
        x_train, y_train = episode['Dtrain']
        phi_train = self.input_transformer(x_train)
        phi_y_train = torch.cat((phi_train, y_train), dim=1)
        phi_test = self.input_transformer(episode['Dtest'])
        n_test = phi_test.size(0)
        y_useless = torch.zeros(n_test, 1)
        y_useless.requires_grad_(False)
        if torch.cuda.is_available():
            y_useless = y_useless.cuda()
        phi_y_test = torch.cat((phi_test, y_useless), dim=1)
        phi_y_train = phi_y_train.expand(n_test, -1, -1)
        phi_y_test = phi_y_test.unsqueeze(1)
        input_ = torch.cat((phi_y_train, phi_y_test), dim=1)

        n = input_.size(0)
        batch_size = 512
        if n > batch_size:
            outs = [self.net(input_[i:i+batch_size])[:, -1, :].contiguous()
                    for i in range(0, n, batch_size)]
            res = torch.cat(outs)
        else:
            res = self.net(input_)[:, -1, :].contiguous()
        return res

    def forward(self, episodes):
        return [self.__forward(episode) for episode in episodes]


class SNAIL(MetaLearnerRegression):
    def __init__(self, input_transformer: ClonableModule, k, arch, loss=mse_loss, lr=0.001):
        super(SNAIL, self).__init__()
        self.lr = lr
        self.loss = loss
        self.network = SNAILNetwork(input_transformer, k, arch)

        if torch.cuda.is_available():
            self.network.cuda()

        optimizer = Adam(self.network.parameters(), lr=self.lr)
        self.model = Model(self.network, optimizer, self.metaloss)