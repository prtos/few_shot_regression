import torch, csv
from pytoune.utils import tensors_to_variables, torch_to_numpy
from pytoune.framework import Model
from pytoune.framework.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, BestModelRestore


def pearsonr(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r = r_num / r_den
    r = max(min(r, 1.0), -1.0)
    return r


def r2_score(y_pred, y_true):
    mean_y_true = torch.mean(y_true)
    ss_tot = torch.sum(torch.pow(y_true.sub(mean_y_true), 2))
    ss_res = torch.sum(torch.pow(y_pred - y_true, 2))
    r2 = 1 - ss_res/ss_tot
    return r2


class CustomCSVLogger(CSVLogger):
    def __init__(self, filename, batch_granularity=False, separator=','):
        super(CustomCSVLogger, self).__init__(filename, batch_granularity, separator)

    def on_train_begin(self, logs):
        metrics = ['loss'] + self.model.metrics_names

        if self.batch_granularity:
            self.fieldnames = ['epoch', 'batch', 'lr']
        else:
            self.fieldnames = ['epoch', 'lr']
        self.fieldnames += metrics
        self.fieldnames += ['val_' + metric for metric in metrics]
        if hasattr(self.model.model, 'l2_krr'):
            self.fieldnames += ['l2_krr']
        self.csvfile = open(self.filename, 'w', newline='')
        self.writer = csv.DictWriter(self.csvfile, fieldnames=self.fieldnames, delimiter=self.separator)
        self.writer.writeheader()
        self.csvfile.flush()

    def on_batch_end(self, batch, logs):
        if self.batch_granularity:
            logs = self._get_logs_without_unknown_keys(logs)
            if hasattr(self.model.model, 'l2_krr'):
                logs.update(dict(l2_krr=self.model.model.l2_krr))
            self.writer.writerow(logs)
            self.csvfile.flush()

    def on_epoch_end(self, epoch, logs):
        logs = self._get_logs_without_unknown_keys(logs)
        if hasattr(self.model.model, 'l2_krr'):
            logs.update(dict(l2_krr=self.model.model.l2_krr))
        self.writer.writerow(dict(logs, lr=self._get_current_learning_rates()))
        self.csvfile.flush()


class MetaLearnerRegression:
    def __init__(self):
        self.model = None
        self.loss = None

    def metaloss(self, y_preds, y_tests):
        return torch.mean(
                torch.stack(
                    [self.loss(y_pred, y_test) for y_pred, y_test in zip(y_preds, y_tests)]
                ))

    def fit(self, metatrain, metavalid, n_epochs=100, steps_per_epoch=100,
            log_filename=None, checkpoint_filename=None):

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='loss', patience=2, factor=1/10, min_lr=1e-6)
        best_model_restore = BestModelRestore()
        callbacks = [early_stopping, reduce_lr, best_model_restore]
        if log_filename:
            logger = CustomCSVLogger(log_filename, batch_granularity=True, separator='\t')
            callbacks += [logger]
        if checkpoint_filename:
            checkpointer = ModelCheckpoint(checkpoint_filename, monitor='val_loss', save_best_only=True)
            callbacks += [checkpointer]

        self.model.fit_generator(metatrain, metavalid,
                                 epochs=n_epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 callbacks=callbacks)
        self.best_valid_loss = early_stopping.best
        return self

    def evaluate(self, metatest):
        scores_r2, scores_pcc, sizes = dict(), dict(), dict()
        self.model.metrics = [pearsonr, r2_score]
        for batch in metatest:
            for (episode, y_test) in batch:
                episode = tensors_to_variables(episode, volatile=True)
                y_pred = self.model.model([episode])[0].data
                r2 = torch_to_numpy(r2_score(y_pred, y_test))
                pcc = torch_to_numpy(pearsonr(y_pred, y_test))
                ep_name = "".join([chr(i) for i in episode['name'].data.cpu().numpy()])
                if ep_name in scores_pcc:
                    scores_pcc[ep_name].append(pcc)
                    scores_r2[ep_name].append(r2)
                else:
                    scores_pcc[ep_name] = [pcc]
                    scores_r2[ep_name] = [r2]
                sizes[ep_name] = y_test.size(0)

        return scores_r2, scores_pcc, sizes

    def load(self, checkpoint_filename):
        self.model.load_weights(checkpoint_filename)