import torch
from pytoune.utils import tensors_to_variables
from pytoune.framework import Model
from pytoune.framework.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


class MetaLearnerRegression:
    def __init__(self):
        self.model = None
        self.loss = None

    def metaloss(self, y_tests, y_preds):
        return torch.mean(
                torch.stack(
                    [self.loss(y_tests[i], y_preds[i]) for i in range(y_tests.size(0))]
                ))

    def fit(self, metatrain, metavalid, n_epochs=100, steps_per_epoch=100,
            log_filename=None, checkpoint_filename=None):

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='loss', patience=2, factor=1/10, min_lr=1e-6)
        callbacks = [early_stopping, reduce_lr]
        if log_filename:
            logger = CSVLogger(log_filename, separator='\t')
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
        for batch in metatest:
            batch = tensors_to_variables(batch, volatile=True)
            preds = self.model.predict(batch)
            for (episode, y_test), y_pred in zip(batch, preds):
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

    def load(self, checkpoint_filename):
        self.model.load_weights(checkpoint_filename)