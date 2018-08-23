import numpy as np
import torch.nn as nn
from torch.nn.utils import weight_norm
from few_shot_regression.utils.feature_extraction.common_modules import *


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCNN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TcnnFeaturesExtractor(ClonableModule):
    def __init__(self, vocab_size, embedding_size, nb_kernels, kernel_size,
                 lmax, dropout=0.2, normalize_features=True):
        super(TcnnFeaturesExtractor, self).__init__()
        self.params = locals()
        del self.params['__class__']
        del self.params['self']
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.nb_kernels = nb_kernels
        self.normalize_kernel = normalize_features
        self.lmax = lmax

        layers = [EmbeddingViaLinear(vocab_size, self.embedding_size),
                  Transpose(1, 2),
                  TCNN(self.embedding_size, [nb_kernels]*np.ceil(np.log2(lmax)), kernel_size)]

        if normalize_features:
            layers.append(UnitNormLayer())

        self.net = nn.Sequential(*layers)

    @property
    def output_dim(self):
        return self.cnn_sizes[-1]

    def forward(self, x):
        return self.net(x)

    def clone(self):
        model = TcnnFeaturesExtractor(**self.params)
        if next(self.parameters()).is_cuda:
            model = model.cuda()
        return model
