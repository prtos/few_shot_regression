from torch.nn import Linear, Sequential, ReLU
from metalearn.feature_extraction.common_modules import *


class FcFeaturesExtractor(ClonableModule):
    def __init__(self, input_size, hidden_sizes, normalize_features=True):
        super(FcFeaturesExtractor, self).__init__()
        self.params = locals()
        del self.params['__class__']
        del self.params['self']
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.normalize_kernel = normalize_features
        layers = []
        in_ = input_size
        for i, out_ in enumerate(hidden_sizes):
            layers.append(Linear(in_, out_))
            if i < len(hidden_sizes) -1:
                layers.append(ReLU())
            in_ = out_

        if normalize_features:
            layers.append(UnitNormLayer())

        self.net = Sequential(*layers)

    @property
    def output_dim(self):
        return self.hidden_sizes[-1] if len(self.hidden_sizes) > 0 else self.input_size

    def forward(self, x):
        return self.net(x)

    def clone(self):
        model = FcFeaturesExtractor(**self.params)
        if next(self.parameters()).is_cuda:
            model = model.cuda()
        return model