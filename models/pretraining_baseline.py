import torch
from torch.nn.functional import normalize,  mse_loss
from torch.optim import Adam, SGD
from pytoune.utils import tensors_to_variables
from .base import MetaLearnerRegression, Model
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


class PretrainBase(MetaLearnerRegression):
    def __init__(self, learner_network, loss=mse_loss, lr=0.001):
        super(PretrainBase, self).__init__()
        self.lr = lr
        self.learner_network = learner_network
        self.loss = loss

        if torch.cuda.is_available():
            self.learner_network.cuda()

        optimizer = Adam(self.learner_network.parameters(), lr=self.lr)
        self.model = Model(self.learner_network, optimizer, self.loss)

    def fit(self, metatrain, metavalid, n_epochs=100, steps_per_epoch=1000,
            log_filename=None, checkpoint_filename=None):
        gtrain = metatrain.full_datapoints_generator()
        gvalid = metavalid.full_datapoints_generator()
        return super(PretrainBase, self).fit(gtrain, gvalid, n_epochs,
                                             steps_per_epoch, log_filename, checkpoint_filename)

    def fine_tune(self, x_train, y_train, lr, n_epochs):
        new_learner = self.learner_network.clone()
        new_learner.load_state_dict(self.learner_network.state_dict())

        if torch.cuda.is_available():
            new_learner.cuda()
        optimizer = Adam(new_learner.parameters(), lr=lr)
        model = Model(new_learner, optimizer, self.loss)

        gen = PretrainBase.make_generator(x_train, y_train)
        model.fit_generator(gen, gen, epochs=n_epochs, steps_per_epoch=1)

        return new_learner

    def evaluate(self, metatest, lr=1e-3, n_epochs=10):
        n = 100
        scores_r2, scores_pcc, sizes = dict(), dict(), dict()
        for batch in metatest:
            batch = tensors_to_variables(batch, volatile=False)
            learners = [self.fine_tune(*episode['Dtrain'], lr, n_epochs) for episode in batch]
            for episode, learner in zip(batch, learners):
                x_test, y_test = episode['Dtest']                
                y_pred = torch.cat([learner(x_test[i:i+n]) if i+n < x_test.size(0) else learner(x_test[i:])
                                    for i in range(0, x_test.size(0), n)], dim=0)
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

    @staticmethod
    def make_generator(x, y, batch_size=-1):
        if batch_size == -1:
            while True:
                yield x, y
        else:
            while True:
                for i in range(0, len(y), batch_size):
                    yield x[i:i+batch_size], y[i:i+batch_size]


