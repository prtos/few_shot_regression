import torch
from torch.nn import Linear, Sequential
from torch.nn.functional import mse_loss
from torch.optim import Adam
from pytoune.framework import Model
from .base import MetaLearnerRegression
from collections import OrderedDict
from few_shot_regression.utils.feature_extraction.common_modules import ClonableModule
from .utils import set_params


class Regressor(ClonableModule):
    def __init__(self, feature_extractor: ClonableModule, output_dim=1):
        super(Regressor, self).__init__()
        self.feature_extractor = feature_extractor
        self.out_dim = output_dim
        output_layer = Linear(self.feature_extractor.output_dim, output_dim)

        self.net = Sequential(self.feature_extractor, output_layer)

    def forward(self, x):
        # print(x.size())
        # print(x.sum())
        # print(self.net(x))
        # exit()
        return self.net(x)

    def clone(self):
        model = Regressor(self.feature_extractor.clone(), self.out_dim)
        if next(self.parameters()).is_cuda:
            model = model.cuda()
        return model


class MAMLNetwork(torch.nn.Module):

    def __init__(self, feature_extractor: ClonableModule, loss=mse_loss, lr_learner=0.02, n_epochs_learner=1):
        """
        In the constructor we instantiate an lstm module
        """
        super(MAMLNetwork, self).__init__()
        self.lr_learner = lr_learner
        self.base_learner = Regressor(feature_extractor.clone(), 1)
        self.n_epochs_learner = n_epochs_learner
        self.loss = loss

    def __forward(self, episode):
        x_train, y_train = episode['Dtrain']
        x_test = episode['Dtest']
        x_train.volatile = False
        learner_network = self.base_learner.clone()

        initial_params = OrderedDict((name, param - 0.0) for (name, param) in self.base_learner.named_parameters())
        set_params(learner_network, initial_params)
        for i in range(self.n_epochs_learner):
            # forward pass with x_train
            output = learner_network(x_train)
            # computation of the loss and the gradients wrt the parameters
            loss = self.loss(output, y_train)
            grads = torch.autograd.grad(loss, learner_network.parameters(), create_graph=True)
            new_weights = OrderedDict((name, param - self.lr_learner*grad)
                                      for ((name, param), grad) in
                                      zip(learner_network.named_parameters(), grads))
            set_params(learner_network, new_weights)

        n = x_test.size(0)
        batch_size = 512
        if n > batch_size:
            outs = [learner_network(x_test[i:i + batch_size])
                    for i in range(0, n, batch_size)]
            res = torch.cat(outs)
        else:
            res = learner_network(x_test)
        return res

    def forward(self, episodes):
        return [self.__forward(episode) for episode in episodes]


class MAML(MetaLearnerRegression):
    def __init__(self, learner_network: ClonableModule, loss=mse_loss, lr=0.001,
                 lr_learner=0.02, n_epochs_learner=1):
        super(MAML, self).__init__()
        self.lr = lr
        self.learner_network = learner_network
        # if isinstance(learner_network, LstmBasedRegressor):
        #     print("Switch torch backend")
        #     torch.backends.cudnn.enabled = False
        self.loss = loss
        self.network = MAMLNetwork(learner_network, loss, lr_learner, n_epochs_learner)

        if torch.cuda.is_available():
            self.network.cuda()

        optimizer = Adam(self.network.parameters(), lr=self.lr)
        self.model = Model(self.network, optimizer, self.metaloss)