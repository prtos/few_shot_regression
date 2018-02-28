import torch
from torch.nn.functional import normalize,  mse_loss
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD
from pytoune.utils import tensors_to_variables
from pytoune.framework.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from .base import MetaModel, MetaLearnerRegression
from .modules import ClonableModule, LstmBasedRegressor
from collections import OrderedDict
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


def set_params(module, new_params, prefix=''):
    module._parameters = OrderedDict((name, new_params[prefix + ('.' if prefix else '') + name])
                                   for name, _ in module._parameters.items())
    for mname, submodule in module.named_children():
        submodule_prefix = prefix + ('.' if prefix else '') + mname
        set_params(submodule, new_params, submodule_prefix)


class MAMLNetwork(torch.nn.Module):

    def __init__(self, learner_network: ClonableModule, loss=mse_loss, lr_learner=0.02, n_epochs_learner=1):
        """
        In the constructor we instantiate an lstm module
        """
        super(MAMLNetwork, self).__init__()
        self.lr_learner = lr_learner
        self.base_learner = learner_network.clone()
        self.n_epochs_learner = n_epochs_learner
        self.loss = loss

    def __forward(self, episode):
        x_train, y_train = episode['Dtrain']
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

        return learner_network

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

        if torch.cuda.is_available() and not isinstance(learner_network, LstmBasedRegressor):
            self.network.cuda()

        optimizer = Adam(self.network.parameters(), lr=self.lr)
        self.model = MetaModel(self.network, optimizer, self.metaloss)

    def metaloss(self, episodes, learners):
        for i, (episode, learner) in enumerate(zip(episodes, learners)):
            x_test, y_test = episode['Dtest']
            y_pred = learner(x_test)
            if i == 0:
                loss = self.loss(y_pred, y_test)
            else:
                loss += self.loss(y_pred, y_test)
        return loss/len(episodes)

    def fit(self, metatrain, metavalid, n_epochs=100, steps_per_epoch=100,
            log_filename=None, checkpoint_filename=None):
        if isinstance(self.learner_network, LstmBasedRegressor):
            metavalid.use_available_gpu = False
            metatrain.use_available_gpu = False
        return super(MAML, self).fit(metatrain, metavalid, n_epochs, steps_per_epoch, log_filename, checkpoint_filename)

    def evaluate(self, metatest):
        if isinstance(self.learner_network, LstmBasedRegressor):
            metatest.use_available_gpu = False
        scores_r2, scores_pcc, sizes = dict(), dict(), dict()
        for batch in metatest:
            batch = tensors_to_variables(batch, volatile=False)
            learners = self.model.predict(batch)
            for episode, learner in zip(batch, learners):
                x_test, y_test = episode['Dtest']
                y_pred = learner(x_test)
                x, y = y_test.data.cpu().numpy().flatten(), y_pred.data.cpu().numpy().flatten()
                r2 = float(r2_score(x, y))
                pcc = float(pearsonr(x, y)[0])
                ep_name = "".join([chr(i) for i in episode['name'].data.cpu().numpy()])
                if ep_name in scores_pcc:
                    scores_pcc[ep_name].append(pcc)
                    scores_r2[ep_name].append(r2)
                else:
                    scores_pcc[ep_name] = [pcc]
                    scores_r2[ep_name] = [r2]
                sizes[ep_name] = y_test.size(0)

        return scores_r2, scores_pcc, sizes

