from torch.nn import Conv1d, Sequential, ReLU, MaxPool1d, BatchNorm1d
from metalearn.feature_extraction.common_modules import *


class Cnn1dFeaturesExtractor(ClonableModule):
    def __init__(self, vocab_size, embedding_size, cnn_sizes, kernel_size, pooling_len=1, dilatation_rate=1,
                 normalize_features=True, lmax=None, use_bn=False, use_residual=False):
        assert type(pooling_len) in [list, int], "pooling_len should be of type int or int list"
        super(Cnn1dFeaturesExtractor, self).__init__()
        self.params = locals()
        del self.params['__class__']
        del self.params['self']
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.cnn_sizes = cnn_sizes
        if type(pooling_len) == int:
            self.pooling_len = [pooling_len] * len(cnn_sizes)
        if type(kernel_size) == int:
            self.kernel_size = [kernel_size] * len(cnn_sizes)
        self.normalize_kernel = normalize_features
        self.dilatation_rate = 1
        self.use_bn = use_bn
        embedding = EmbeddingViaLinear(vocab_size, self.embedding_size)
        layers = []
        layers.append(Transpose(1, 2))
        in_channels = [self.embedding_size] + cnn_sizes[:-1]

        """
                      L_{out} = \left\lfloor \frac{L_{in} + 2 * \text{padding} - \text{dilation}
                            * (\text{kernel_size} - 1) - 1}{\text{stride}} + 1\right\rfloor
        """
        for i, (in_channel, out_channel, ksize, l_pool) in \
                enumerate(zip(in_channels, cnn_sizes, self.kernel_size, self.pooling_len)):
            pad = ((dilatation_rate**i) * (ksize - 1) + 1) // 2
            layers.append(Conv1d(in_channel, out_channel, padding=pad,
                                 kernel_size=ksize, dilation=dilatation_rate**i))
            if use_bn and i < len(in_channels) - 1:
                layers.append(BatchNorm1d(out_channel))
            if i < len(in_channels) - 1:
                layers.append(ReLU())
            if l_pool > 1:
                layers.append(MaxPool1d(pooling_len))
        layers.append(Transpose(1, 2))
        if lmax is None:
            layers.append(GlobalAvgPool1d())
        else:
            n = lmax * self.cnn_sizes[-1]
        if normalize_features:
            layers.append(UnitNormLayer())

        self.net = Sequential(embedding, *layers)

    @property
    def output_dim(self):
        return self.cnn_sizes[-1]

    def forward(self, x):
        # print(x)
        return self.net(x)

    def clone(self):
        model = Cnn1dFeaturesExtractor(**self.params)
        if next(self.parameters()).is_cuda:
            model = model.cuda()
        return model


if __name__ == '__main__':
    x = torch.Tensor(32, 80, 4)
    model = Cnn1dFeaturesExtractor(4, 50, cnn_sizes=[25] * 4, kernel_size=5, dilatation_rate=2, pooling_len=2)
    y = model(x)
    print(y.size())
