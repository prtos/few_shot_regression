import csv
import numpy as np
from tensorboardX import SummaryWriter
from pytoune.utils import torch_to_numpy
from pytoune.framework.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, \
    BestModelRestore, TensorBoardLogger
from few_shot_regression.utils.metric import *
from torch.nn.functional import mse_loss
from pytoune.framework import warning_settings
warning_settings['batch_size'] = 'ignore'


class Krr_CSVLogger(CSVLogger):
    def __init__(self, filename, separator=','):
        super(Krr_CSVLogger, self).__init__(filename, True, separator)

    def on_train_begin(self, logs):
        metrics = ['loss'] + self.model.metrics_names

        self.fieldnames = ['epoch', 'batch', 'lr', 'l2', 'phi_norm', 'w_norm', 'y_mean', 'y_std']
        self.fieldnames += metrics
        self.fieldnames += ['val_' + metric for metric in metrics]
        self.csvfile = open(self.filename, 'w', newline='')
        self.writer = csv.DictWriter(self.csvfile, fieldnames=self.fieldnames, delimiter=self.separator)
        self.writer.writeheader()
        self.csvfile.flush()

    def _get_logs_without_unknown_keys(self, logs):
        if hasattr(self.model.network, 'attr_watched'):
            temp = getattr(self.model.model, 'attr_watched', {})
            res = {k: temp[k]
                   for k in self.fieldnames if temp.get(k)}
        else:
            res = {}
        res.update({k: logs[k] for k in self.fieldnames if logs.get(k)})
        return res


class MetaLearnerRegression:
    def __init__(self):
        self.model = None
        self.network = None
        self.is_fitted = False

    def metaloss(self, y_preds, y_tests):
        return torch.mean(torch.stack([mse_loss(y_pred, y_test) for y_pred, y_test in zip(y_preds, y_tests)]))

    def train_test_split(self, dataset, test_size):
        return dataset.train_test_split(test_size=test_size)

    def fit(self, metatrain, valid_size=0.25, n_epochs=100, steps_per_epoch=100,
            batch_size=32, log_filename=None, checkpoint_filename=None, tboard_folder=None):
        meta_train, meta_valid = self.train_test_split(metatrain, valid_size)
        print("Number of train steps:", len(meta_train))
        print("Number of valid steps:", len(meta_valid))

        callbacks = []
        early_stopping = EarlyStopping(patience=10, verbose=False)
        callbacks.append(early_stopping)

        reduce_lr = ReduceLROnPlateau(patience=5, factor=1/2, min_lr=1e-6)
        best_model_restore = BestModelRestore()
        callbacks += [reduce_lr, best_model_restore]
        if log_filename:
            logger = CSVLogger(log_filename, batch_granularity=False, separator='\t')
            callbacks += [logger]
            if hasattr(self.network, 'attr_watched'):
                callbacks += [Krr_CSVLogger(log_filename + 'krr', separator='\t')]
        if checkpoint_filename:
            checkpointer = ModelCheckpoint(checkpoint_filename, monitor='val_loss', save_best_only=True,
                                           temporary_filename=checkpoint_filename+'temp')
            callbacks += [checkpointer]

        if tboard_folder is not None:
            writer = SummaryWriter(tboard_folder)
            if hasattr(self.network, 'set_writer'):
                self.network.set_writer(writer)
            tboard = TensorBoardLogger(writer)
            callbacks.append(tboard)

        self.model.fit_generator(meta_train, meta_valid,
                                 epochs=n_epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_steps=None,
                                 callbacks=callbacks,
                                 verbose=True)
        self.is_fitted = True
        return self

    def evaluate(self, metatest, metrics=[]):
        assert len(metrics) >= 1, "There should be at least one valid metric in the list of metrics "
        metric_names = [f.__name__ for f in metrics]
        metrics_per_dataset = {metric_name: {} for metric_name, metric in zip(metric_names, metrics)}
        metrics_per_dataset["size"] = dict()
        for batch in metatest:
            for episode, y_test in zip(*batch):
                ep_name = "".join([chr(i) for i in episode['name'].data.cpu().numpy()])
                y_pred = self.model.model([episode])[0]
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0].data
                ep_name_is_new = (ep_name not in metrics_per_dataset["size"])
                for metric_name, metric in zip(metric_names, metrics):
                    m_value = torch_to_numpy(metric(y_pred, y_test))
                    if ep_name_is_new:
                        metrics_per_dataset[metric_name][ep_name] = [m_value]
                    else:
                        metrics_per_dataset[metric_name][ep_name].append(m_value)
                metrics_per_dataset['size'][ep_name] = y_test.size(0)

        return metrics_per_dataset

    def load(self, checkpoint_filename):
        self.model.load_weights(checkpoint_filename)
        self.is_fitted = True