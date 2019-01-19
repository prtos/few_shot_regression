import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from pytoune.utils import torch_to_numpy
from pytoune.framework.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, \
    BestModelRestore, TensorBoardLogger, DelayCallback
from metalearn.feature_extraction.factory import FeaturesExtractorFactory
from torch.nn.functional import mse_loss
from torch.utils.data import random_split
from pytoune.framework import warning_settings, Model
from metalearn.models.utils import to_unit, to_numpy_vec
warning_settings['batch_size'] = 'ignore'


def get_optimizer_cls(optimizer):
    OPTIMIZERS = {k.lower(): v for k, v in vars(torch.optim).items() 
                  if not k.startswith('__')}
    return OPTIMIZERS[optimizer]


class MetaNetwork(torch.nn.Module):
    @property
    def return_var(self):
        return False

    def __forward(self, episode):
        raise NotImplementedError()

    def forward(self, episodes):
        return [self.__forward(episode) for episode in episodes]


class MetaLearnerRegression(Model):
    def __init__(self, network, optimizer, lr, weight_decay):
        if torch.cuda.is_available():
            network.cuda()
        opt = get_optimizer_cls(optimizer)(network.parameters(), lr=lr, weight_decay=weight_decay)
        super(MetaLearnerRegression, self).__init__(network, opt, self._on_batch_end)
        self.is_fitted = False
        self.writer = None
        self.train_step = 0
        self.test_step = 0
        self.is_eval = False

    def _compute_aux_return_loss(self, y_preds, y_tests):
        loss = torch.mean(torch.stack([mse_loss(y_pred, y_test) 
                for y_pred, y_test in zip(y_preds, y_tests)]))
        return loss, dict(mse=loss)

    def _on_batch_end(self, y_preds, y_tests):
        loss, scalars = self._compute_aux_return_loss(y_preds, y_tests)

        t = self.train_step if self.model.training else self.test_step
        tag = 'train' if self.model.training else 'val'

        if self.writer is not None:
            tag = 'train' if self.model.training else 'valid'
            for k, v in scalars.items():
                self.writer.add_scalar(f'{tag}/{k}', to_unit(v), t)
        if torch.isnan(loss).any():
            raise Exception(f'{self.__class__.__name__}: Loss goes NaN')
        
        if self.model.training:
            self.train_step += 1
        else:
            self.test_step += 1
        return loss

    def fit(self, meta_train, meta_valid, n_epochs=100, steps_per_epoch=100,
            log_filename=None, checkpoint_filename=None, tboard_folder=None):
        if hasattr(self.model, 'is_eval'):
            self.model.is_eval = False
        self.is_eval = False
        self.steps_per_epoch = steps_per_epoch
        callbacks = [EarlyStopping(patience=5, verbose=False),
                     ReduceLROnPlateau(patience=5, factor=1/2, min_lr=1e-6, verbose=True),
                     BestModelRestore()]
        if log_filename:
            callbacks += [CSVLogger(log_filename, batch_granularity=False, separator='\t')]
        if checkpoint_filename:
            callbacks += [ModelCheckpoint(checkpoint_filename, monitor='val_loss', save_best_only=True,
                                          temporary_filename=checkpoint_filename+'temp')]

        if tboard_folder is not None:
            self.writer = SummaryWriter(tboard_folder)

        self.fit_generator(meta_train, meta_valid,
                            epochs=n_epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=steps_per_epoch,
                            callbacks=callbacks,
                            verbose=True)
        self.is_fitted = True
        return self

    def plot_harmonics(self, episode, step, tag=''):
        preds = self.model([episode])[0]
        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        plt.figure(dpi=60)
        x_train, y_train = episode['Dtrain']
        x_test, y_test = episode['Dtest']
        x_test = to_numpy_vec(x_test)
        x_order = np.argsort(x_test)
        x_test = x_test[x_order]
        y_test = to_numpy_vec(y_test)[x_order]
        plt.plot(to_numpy_vec(x_train), to_numpy_vec(y_train), 'ro')
        plt.plot(x_test, y_test, '-')
        for pred in preds:
            if isinstance(pred, (list, tuple)):
                y_mean, y_var = pred[:2]
            else:
                y_mean, y_var = pred, None
            y_pred = to_numpy_vec(y_mean)[x_order]
            plt.plot(x_test, y_pred, '-')
            if y_var is not None:
                y_var = to_numpy_vec(y_var)[x_order]
                plt.fill_between(x_test, y_pred - y_var, y_pred + y_var)

        self.writer.add_figure(tag=tag, figure=plt.gcf(), close=True, global_step=step)

    def evaluate(self, metatest, metrics=[mse_loss], tboard_folder=None):
        if hasattr(self.model, 'is_eval'):
            self.model.is_eval = True
        self.eval = True
        assert len(metrics) >= 1, "There should be at least one valid metric in the list of metrics "
        if tboard_folder is not None:
            print('here.....\n')
            self.writer = SummaryWriter(tboard_folder)
        metrics_per_dataset = {metric.__name__: {} for metric in metrics}
        metrics_per_dataset["size"] = dict()
        it = 0
        for batch in metatest:
            episodes = batch[0]
            y_preds = self.model(episodes)
            y_tests = batch[1]
            for episode, y_test, y_pred in zip(episodes, y_tests, y_preds):
                if self.model.return_var:
                    y_pred_mean, y_pred_std = y_pred
                else:
                    y_pred_mean = y_pred
                ep_idx = episode['idx']
                is_one_dim_input = (episode['Dtrain'][0].size(1) == 1)
                if is_one_dim_input and np.random.binomial(1, 0.1, 1)[0] == 1 and it <= 5000:
                    it += 1
                    self.plot_harmonics(episode, tag='test'+str(ep_idx), step=it)
                ep_name_is_new = (ep_idx not in metrics_per_dataset["size"])
                for metric in metrics:
                    m_value = to_unit(metric(y_pred_mean, y_test))
                    if ep_name_is_new:
                        metrics_per_dataset[metric.__name__][ep_idx] = [m_value]
                    else:
                        metrics_per_dataset[metric.__name__][ep_idx].append(m_value)
                metrics_per_dataset['size'][ep_idx] = y_test.size(0)

        return metrics_per_dataset

    def load(self, checkpoint_filename):
        self.load_weights(checkpoint_filename)
        self.is_fitted = True

    @staticmethod
    def static_load(checkpoint_filename, optimizer, loss_function, metrics=[]):
        model = torch.load(checkpoint_filename)
        return MetaLearnerRegression(model, optimizer, loss_function, metrics)