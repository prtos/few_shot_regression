import torch, csv
import numpy as np
import subprocess
from torch.autograd import Variable
from torch.nn.functional import mse_loss
from pytoune.utils import tensors_to_variables, torch_to_numpy
from pytoune.framework import Model
from pytoune.framework.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, BestModelRestore


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    # nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    # gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory


def create_var(t):
    v = Variable(t)
    if torch.cuda.is_available():
            v = v.cuda()
    return v


def pearsonr(x, y):
    x, y = x.view(-1), y.view(-1)
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x - mean_x
    ym = y - mean_y
    r_num = torch.sum(xm * ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r = r_num / (r_den + 1e-8)
    r = max(min(r, 1.0), -1.0)
    return r


def r2_score(y_pred, y_true):
    y_pred, y_true = y_pred.view(-1), y_true.view(-1)
    mean_y_true = torch.mean(y_true)
    ss_tot = torch.sum(torch.pow(y_true.sub(mean_y_true), 2))
    ss_res = torch.sum(torch.pow(y_pred - y_true, 2))
    r2 = 1 - (ss_res/(ss_tot + 1e-8))
    return r2


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
        if hasattr(self.model.model, 'attr_watched'):
            temp = getattr(self.model.model, 'attr_watched', {})
            res = {k: temp[k]
                   for k in self.fieldnames if temp.get(k)}
        else:
            res = {}
        res.update({k:logs[k] for k in self.fieldnames if logs.get(k)})
        return res


class MetaLearnerRegression:
    def __init__(self):
        self.model = None
        self.network = None
        self.loss = None

    def metaloss(self, y_preds, y_tests):
        return torch.mean(
                torch.stack(
                    [self.loss(y_pred, y_test) for y_pred, y_test in zip(y_preds, y_tests)]
                ))

    def train_test_split(self, dataset, test_size):
        return dataset.train_test_split(test_size=test_size)

    def fit(self, metatrain, valid_size=0.25, n_epochs=100, steps_per_epoch=100,
            max_episodes=None, batch_size=32,
            log_filename=None, checkpoint_filename=None):
        meta_train, meta_valid = self.train_test_split(metatrain, valid_size)
        callbacks = []
        if max_episodes is not None:
            n_epochs = int(np.ceil(max_episodes/(steps_per_epoch*batch_size*1.)))
        else:
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
            callbacks.append(early_stopping)

        reduce_lr = ReduceLROnPlateau(monitor='loss', patience=2, factor=1/10, min_lr=1e-6)
        best_model_restore = BestModelRestore()
        callbacks += [reduce_lr, best_model_restore]
        if log_filename:
            logger = CSVLogger(log_filename, batch_granularity=False, separator='\t')
            callbacks += [logger]
            if hasattr(self.network, 'attr_watched'):
                callbacks += [Krr_CSVLogger(log_filename + 'krr', separator='\t')]
        if checkpoint_filename:
            checkpointer = ModelCheckpoint(checkpoint_filename, monitor='val_loss', save_best_only=True)
            callbacks += [checkpointer]

        self.model.fit_generator(meta_train, meta_valid,
                                 epochs=n_epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 callbacks=callbacks)
        return self

    def evaluate(self, metatest):
        # gpu_mem = get_gpu_memory_map()
        # print('memory usage', gpu_mem)
        # print('memory usage = {} Gb', gpu_mem[0]/1000.0)
        scores_mse, scores_r2, scores_pcc, sizes = dict(), dict(), dict(), dict()
        self.model.metrics = [pearsonr, r2_score]
        for batch in metatest:
            for episode, y_test in batch:
                episode = tensors_to_variables(episode, volatile=True)
                try:
                    print(y_test.size(0))
                    y_pred = self.model.model([episode])[0].data
                except MemoryError:
                    print('Memory error for test size', y_test.size(0))
                    continue
                mse = torch_to_numpy(torch.mean(torch.pow(y_pred.sub(y_test), 2)))
                r2 = torch_to_numpy(r2_score(y_pred, y_test))
                pcc = torch_to_numpy(pearsonr(y_pred, y_test))
                ep_name = "".join([chr(i) for i in episode['name'].data.cpu().numpy()])
                if ep_name in scores_pcc:
                    scores_pcc[ep_name].append(pcc)
                    scores_r2[ep_name].append(r2)
                    scores_mse[ep_name].append(mse)
                else:
                    scores_pcc[ep_name] = [pcc]
                    scores_r2[ep_name] = [r2]
                    scores_mse[ep_name] = [mse]
                sizes[ep_name] = y_test.size(0)

        return scores_mse, scores_r2, scores_pcc, sizes

    def load(self, checkpoint_filename):
        self.model.load_weights(checkpoint_filename)