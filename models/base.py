from pytoune.framework import *
from pytoune.utils import tensors_to_variables
from pytoune.framework.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


class MetaModel(Model):
    def __init__(self, model, optimizer, loss_function, metrics=[]):
        super(MetaModel, self).__init__(model, optimizer, loss_function, metrics)

    def fit_generator(self, train_generator, valid_generator=None, epochs=1000, steps_per_epoch=None, validation_steps=None, verbose=True, callbacks=[]):

        if verbose:
            callbacks = [ProgressionCallback()] + callbacks
        callback_list = CallbackList(callbacks)
        callback_list.set_model(self)

        if validation_steps is None:
            if hasattr(valid_generator, '__len__'):
                validation_steps = len(valid_generator)
            elif steps_per_epoch is not None:
                validation_steps = steps_per_epoch
            else:
                raise ValueError("Invalid 'validation_steps' value. Either a value for 'validation_steps' or 'steps_per_epoch' must be provided, or 'valid_generator' must provide a '__len__' method.")
        if steps_per_epoch is None:
            steps_per_epoch = len(train_generator)
        params = {'epochs': epochs, 'steps': steps_per_epoch}
        callback_list.set_params(params)

        epoch_logs = []
        self.stop_training = False
        callback_list.on_train_begin({})
        for epoch in range(1, epochs + 1):
            callback_list.on_epoch_begin(epoch, {})
            losses_sum = 0.
            metrics_sum = np.zeros(len(self.metrics))
            times_sum = 0.

            self.model.train(True)
            train_iterator = iter(train_generator)
            for step in range(1, steps_per_epoch + 1):
                callback_list.on_batch_begin(step, {})

                self.model.zero_grad()
                loss_tensor, metrics_tensors = self._run_step(train_iterator)

                loss_tensor.backward()
                self.optimizer.step()

                loss, metrics = self._loss_and_metrics_tensors_to_numpy(loss_tensor, metrics_tensors)
                losses_sum += loss
                metrics_sum += metrics

                metrics_dict = dict(zip(self.metrics_names, metrics))
                batch_logs = {'batch': step, 'loss': loss, **metrics_dict}
                callback_list.on_batch_end(step, batch_logs)

            val_dict = {}
            if valid_generator is not None:
                self.model.eval()
                val_loss, val_metrics = self._validate(valid_generator, validation_steps)
                val_metrics_dict = {'val_' + metric_name:metric for metric_name, metric in zip(self.metrics_names, val_metrics)}
                val_dict = {'val_loss': val_loss, **val_metrics_dict}

            losses_mean = losses_sum / steps_per_epoch
            metrics_mean = metrics_sum / steps_per_epoch
            metrics_dict = dict(zip(self.metrics_names, metrics_mean))
            epoch_log = {'epoch': epoch, 'loss': losses_mean, **metrics_dict, **val_dict}
            callback_list.on_epoch_end(epoch, epoch_log)

            epoch_logs.append(epoch_log)

            if self.stop_training:
                break

        callback_list.on_train_end({})

        return epoch_logs

    def predict(self, x):
        """
        Returns the tensor of the predictions of the network given a tensor for
        a batch of samples.

        Args:
            x (torch.Tensor): A batch of samples.

        Returns:
            The tensor of the predictions of the network given a tensor for
            the batch of samples ``x``.
        """
        self.model.eval()
        x = tensors_to_variables(x, volatile=True)
        return self.model(x)

    def _validate(self, valid_generator, validation_steps):
        losses_list = np.zeros(validation_steps)
        metrics_list = np.zeros((validation_steps,len(self.metrics)))
        valid_iterator = iter(valid_generator)
        for step in range(validation_steps):
            loss_tensor, metrics_tensors = self._run_step(valid_iterator)
            loss, metrics = self._loss_and_metrics_tensors_to_numpy(loss_tensor, metrics_tensors)
            losses_list[step] = loss
            metrics_list[step] = metrics
        return losses_list.mean(), metrics_list.mean(0)

    def _run_step(self, iterator):
        episode = next(iterator)
        episode = tensors_to_variables(episode)
        learner = self.model(episode)
        loss_tensor = self.loss_function(episode, learner)
        metrics = self._compute_metrics(episode, learner)
        return loss_tensor, metrics


class MetaLearnerRegression:
    def __init__(self):
        self.model = None

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

    def load(self, checkpoint_filename):
        self.model.load_weights(checkpoint_filename)