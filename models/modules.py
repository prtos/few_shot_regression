import torch
from torch.nn.functional import normalize
from torch.nn import LSTM, Linear, Conv1d, Sequential, ReLU, Dropout, MaxPool1d, Module
from torch.nn.utils.rnn import pack_padded_sequence


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


class Cnn1dFeaturesExtractor(Module):
    def __init__(self, vocab_size, embedding_size, cnn_sizes, kernel_size, pooling_len=1, dilatation_rate=1,
                 normalize_features=True, lmax=None):
        assert type(pooling_len) in [list, int], "pooling_len should be of type int or int list"
        super(Cnn1dFeaturesExtractor, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.cnn_sizes = cnn_sizes
        if type(pooling_len) == int:
            self.pooling_len = [pooling_len]*len(cnn_sizes)
        if type(kernel_size) == int:
            self.kernel_size = [kernel_size]*len(cnn_sizes)
        self.batch_first = True
        self.normalize_kernel = normalize_features
        self.dilatation_rate = 1
        embedding = EmbeddingViaLinear(vocab_size, self.embedding_size)
        layers = []
        layers.append(Transpose(1, 2))
        in_channels = [self.embedding_size] + cnn_sizes[:-1]

        for i, (in_channel, out_channel, ksize, l_pool) in \
                enumerate(zip(in_channels, cnn_sizes, self.kernel_size, self.pooling_len)):
            pad = ((dilatation_rate**i) * (ksize - 1) + 1) // 2
            layers.append(Conv1d(in_channel, out_channel, padding=pad,
                                 kernel_size=ksize, dilation=dilatation_rate**i))
            layers.append(ReLU())
            if l_pool > 1:
                layers.append(MaxPool1d(pooling_len))
        layers.append(Transpose(1, 2))
        # if lmax is None:
        #     layers.append(GlobalAvgPool1d())
        # else:
        #     n = lmax * self.cnn_sizes[-1]
        if normalize_features:
            layers.append(UnitNormLayer())

        self.net = Sequential(embedding, *layers)

    @property
    def output_dim(self):
        return self.cnn_sizes[-1]

    def forward(self, x):
        print(x.size())
        print(self.net(x).size())
        exit()
        return self.net(x)


class LstmFeaturesExtractor(Module):
    def __init__(self, vocab_size, embedding_size, lstm_hidden_size, nb_lstm_layers,
                 bidirectional=True, normalize_features=True):
        super(LstmFeaturesExtractor, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lstm_hidden_size = lstm_hidden_size
        self.nb_lstm_layers = nb_lstm_layers
        self.batch_first = True
        self.concat_lstms = True
        self.normalize_output = normalize_features
        self.bidirectional = bidirectional
        self.embedding = EmbeddingViaLinear(vocab_size, self.embedding_size)
        self.lstm = LSTM(input_size=self.embedding_size,
                                        num_layers=self.nb_lstm_layers,
                                        hidden_size=self.lstm_hidden_size,
                                        bidirectional=self.bidirectional,
                                        batch_first=self.batch_first)
        if self.concat_lstms:
            self.od = lstm_hidden_size * nb_lstm_layers * (2 if self.bidirectional else 1)
        else:
            self.od = lstm_hidden_size

        if self.normalize_output:
            self.norm_layer = UnitNormLayer()

    @property
    def output_dim(self):
        return self.od

    def forward(self, x):
        x_ = x.view(x.size(0), -1)
        lengths = (x_ > 0).long().sum(1)
        lengths, perm_idx = lengths.sort(0, descending=True)
        _, rev_perm_idx = perm_idx.sort(0)

        x = x[perm_idx]
        batch_size = x.size(0)
        # h_0 = Variable(torch.zeros((x_train.size(0), self.nb_layers * self.num_directions, self.hidden_size)))
        embedding = self.embedding(x)
        packed_x_train = pack_padded_sequence(embedding, lengths.data.cpu().numpy(), batch_first=self.batch_first)
        packed_output, (hidden, _) = self.lstm(packed_x_train)
        # output, _ = pad_packed_sequence(packed_output, batch_first=self.batch_first)
        if self.concat_lstms:
            hidden = hidden.transpose(0, 1).contiguous()
            phis = hidden.view(batch_size, -1)
        else:
            phis = hidden[-1]

        if self.normalize_output:
            phis = self.norm_layer(phis)

        return phis[rev_perm_idx]


class LstmBasedRegressor(ClonableModule):
    def __init__(self, vocab_size, embedding_size, lstm_hidden_size, nb_lstm_layers, bidirectional=True,
                 normalize_features=True, output_dim=1):
        super(LstmBasedRegressor, self).__init__()
        self.params = locals()
        del self.params['__class__']
        del self.params['self']
        features_extraction = LstmFeaturesExtractor(vocab_size, embedding_size, lstm_hidden_size, nb_lstm_layers,
                                                    bidirectional, normalize_features)
        output_layer = Linear(features_extraction.output_dim, output_dim)

        self.net = Sequential(features_extraction, output_layer)

    def forward(self, x):
        return self.net(x)

    def clone(self):
        return LstmBasedRegressor(**self.params)


class Cnn1dBasedRegressor(ClonableModule):
    def __init__(self, vocab_size, embedding_size, cnn_sizes, kernel_size, pooling_len=1, dilatation_rate=1,
                 normalize_features=True, output_dim=1):
        super(Cnn1dBasedRegressor, self).__init__()
        self.params = locals()
        del self.params['__class__']
        del self.params['self']
        features_extraction = Cnn1dFeaturesExtractor(vocab_size, embedding_size, cnn_sizes,
                                                     kernel_size, pooling_len, dilatation_rate, normalize_features)
        output_layer = Linear(features_extraction.output_dim, output_dim)

        self.net = Sequential(features_extraction, output_layer)

    def forward(self, x):
        return self.net(x)

    def clone(self):
        return Cnn1dBasedRegressor(**self.params)


if __name__ == '__main__':
    LstmBasedRegressor(20, 20, 21, 2).clone()