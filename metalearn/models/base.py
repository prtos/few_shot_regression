import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from pytoune.utils import torch_to_numpy
from pytoune.framework.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, \
    BestModelRestore, TensorBoardLogger, DelayCallback
from torch.nn.functional import mse_loss
from pytoune.framework import warning_settings
warning_settings['batch_size'] = 'ignore'


def to_numpy_vec(x):
    return x.data.cpu().numpy().flatten()


def write_episode_and_preds(episode, res, writer, step, tag=''):
    is_a_collection = isinstance(res, list)
    res = res if is_a_collection else [res]
    plt.figure()
    x_train, y_train = episode['Dtrain']
    x_test, y_test = episode['Dtest']
    x_test = to_numpy_vec(x_test)
    x_order = np.argsort(x_test)
    x_test = x_test[x_order]
    y_test = to_numpy_vec(y_test)[x_order]
    plt.plot(to_numpy_vec(x_train), to_numpy_vec(y_train), 'bo')
    plt.plot(x_test, y_test, '-')
    for preds in res:
        y_pred, y_var = (preds[:, 0], preds[:, 1]) if preds.size(1) == 2 else (preds, None)
        y_pred = to_numpy_vec(y_pred)[x_order]
        plt.plot(x_test, y_pred, '-', color='gray')
        if y_var is not None:
            y_var = to_numpy_vec(y_var)[x_order]
            plt.fill_between(x_test, y_pred - y_var, y_pred + y_var, color='gray', alpha=0.2)
    # buf = io.BytesIO()
    # plt.savefig(buf, format='jpeg')
    # buf.seek(0)
    # img = ToTensor()(PIL.Image.open(buf)).unsqueeze(0)
    # writer.add_image(tag='test_functions', img_tensor=img, global_step=1)
    # plt.close()

    writer.add_figure(tag=tag, figure=plt.gcf(), close=True, global_step=step)


class MetaLearnerRegression:
    def __init__(self):
        self.model = None
        self.network = None
        self.is_fitted = False
        self.writer = None

    def metaloss(self, y_preds, y_tests):
        return torch.mean(torch.stack([mse_loss(y_pred, y_test) for y_pred, y_test in zip(y_preds, y_tests)]))

    def train_test_split(self, dataset, test_size):
        return dataset.train_test_split(test_size=test_size)

    def fit(self, metatrain, valid_size=0.25, n_epochs=100, steps_per_epoch=100,
            batch_size=32, log_filename=None, checkpoint_filename=None, tboard_folder=None):
        meta_train, meta_valid = self.train_test_split(metatrain, valid_size)
        print("Number of train steps:", len(meta_train))
        print("Number of valid steps:", len(meta_valid))

        callbacks = [# EarlyStopping(patience=10, verbose=False),)
                     ReduceLROnPlateau(patience=5, factor=1/2, min_lr=1e-6),
                     BestModelRestore()]
        if log_filename:
            callbacks += [CSVLogger(log_filename, batch_granularity=False, separator='\t')]
        if checkpoint_filename:
            callbacks += [ModelCheckpoint(checkpoint_filename, monitor='val_loss', save_best_only=True,
                                          temporary_filename=checkpoint_filename+'temp')]

        if tboard_folder is not None:
            self.writer = SummaryWriter(tboard_folder)
            if hasattr(self.network, 'set_writer'):
                self.network.set_writer(self.writer)
            callbacks += [TensorBoardLogger(self.writer)]

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
        it = 0
        for batch in metatest:
            for episode, y_test in zip(*batch):
                it +=1
                ep_name = os.path.basename("".join([chr(i) for i in episode['name'].data.cpu().numpy()]))
                res = self.model.model([episode])[0]
                is_one_dim_input = (episode['Dtrain'][0].size(1) == 1)
                if is_one_dim_input:
                    write_episode_and_preds(episode, res, self.writer, tag=ep_name, step=it)
                y_pred = res[0] if isinstance(res, list) else res
                if y_pred.size(1) > 1:  # y_pred contains the predictive mean and variance
                    y_pred = y_pred[:, 0]
                if isinstance(y_pred, tuple):   # y_pred contains other e.g. mean or var of task representation
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