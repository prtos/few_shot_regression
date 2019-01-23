"""
Train support-based models.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import warnings
import numpy as np
import tensorflow as tf
import sys
import time
from deepchem.models import Model
from deepchem.data import pad_batch
from deepchem.data import NumpyDataset
from deepchem.models.tf_new_models.graph_topology import merge_dicts
from deepchem.nn import model_ops
from metalearn.low_data.data_loader import SupportGenerator, EpisodeGenerator
from collections import defaultdict as ddict
from tensorboardX import SummaryWriter


class MetaGraphRegressor(Model):

    def __init__(self, model, test_size=10,
                 support_size=10,
                 learning_rate=.001,
                 tboard=".logs/",
                 **kwargs):
        """Builds a support-based Regressor in deepchem. 
        This is trying to extend the work in 
        See https://arxiv.org/pdf/1606.04080v1.pdf for regression tasks,
        And is based on the old TensorGraph runner of deepchem v1.3.deepchem

        Parameters
        ----------
        sess: tf.Session
            Session for this model
        model: base model for predictions 
            Contains core layers in model. 
        n_pos: int
            Number of positive examples in support.
        n_neg: int
            Number of negative examples in support.
        """
        self.model = model
        self.sess = tf.Session(graph=self.model.graph)
        self.test_size = test_size
        self.support_size = support_size
        self.writer = SummaryWriter(tboard)
        self.learning_rate = learning_rate
        self.epsilon = 1e-7

        with self.model.graph.as_default():
            self.add_placeholders()
            self.pred_op, self.loss_op = self.add_training_loss()
            # Get train function
            self.train_op = self.get_training_op(self.loss_op)

            # Initialize
            self.init_fn = tf.global_variables_initializer()
            self.sess.run(self.init_fn)

    def get_training_op(self, loss):
        """Attaches an optimizer to the graph."""
        opt = tf.train.AdamOptimizer(self.learning_rate)
        return opt.minimize(self.loss_op, name="train")

    def add_placeholders(self):
        self.test_label_placeholder = tf.placeholder(
            dtype='float32', shape=(self.test_size, 1), name="label_placeholder")
        self.test_weight_placeholder = tf.placeholder(
            dtype='float32',
            shape=(self.test_size, 1),
            name="weight_placeholder")

        self.support_label_placeholder = tf.placeholder(
            dtype='float32',
            shape=[self.support_size],
            name="support_label_placeholder")
        self.phase = tf.placeholder(dtype='bool', name='keras_learning_phase')
        # DEBUG

    def construct_feed_dict(self, test, support, training=True, add_phase=False):
        """Constructs tensorflow feed from test/support sets."""
        # Generate dictionary elements for support
        feed_dict = (
            self.model.support_graph_topology.batch_to_feed_dict(support.X))
        feed_dict[self.support_label_placeholder] = np.squeeze(support.y)
        # Get graph information for test
        batch_topo_dict = (
            self.model.test_graph_topology.batch_to_feed_dict(test.X))
        feed_dict = merge_dicts([batch_topo_dict, feed_dict])
        # Generate dictionary elements for test
        feed_dict[self.test_label_placeholder] = test.y
        feed_dict[self.test_weight_placeholder] = test.w

        if add_phase:
            feed_dict[self.phase] = training
        return feed_dict

    def fit(self, dataset, n_episodes_per_epoch=500, nb_epochs=100, log_every=50, **kwargs):
        """Fits model on dataset using cached supports.

        For each epoch, sample n_episodes_per_epoch (support, test) pairs and does
        gradient descent.

        Parameters
        ----------
        dataset: dc.data.Dataset
            Dataset to fit model on.
        nb_epochs: int, optional
            number of epochs of training.
        n_episodes_per_epoch: int, optional
            Number of (support, test) pairs to sample and train on per step.
        log_every: int, optional
            Displays info every this number of samples
        """
        time_start = time.time()
        # Perform the optimization
        n_tasks = len(dataset.get_task_list())
        n_test = self.test_size
        n_train = self.support_size

        feed_total, run_total = 0, 0
        ep_ind = 0
        for epoch in range(nb_epochs):
            # Create different support sets
            episode_generator = EpisodeGenerator(
                dataset, n_train, n_test, n_episodes_per_epoch)
            recent_losses = []

            for ind, (task, support, test) in enumerate(episode_generator):
                if ind % log_every == 0:
                    print("Epoch %d, step %d from task %s" %
                          (epoch, ind, str(task)))
                # Get batch to try it out on
                feed_start = time.time()
                feed_dict = self.construct_feed_dict(test, support)
                feed_end = time.time()
                feed_total += (feed_end - feed_start)
                # Train on support set, batch pair
                run_start = time.time()
                _, loss = self.sess.run(
                    [self.train_op, self.loss_op], feed_dict=feed_dict)
                run_end = time.time()
                run_total += (run_end - run_start)
                recent_losses.append(loss)
                self.writer.add_scalar('epi_loss', loss, ind)
                if ind % log_every == 0 and ind > 0:
                    past_ind = ind - log_every
                    mean_loss = np.mean(recent_losses[past_ind:ind])
                    print("\tmean loss after %d steps is %s" %
                          (log_every, str(mean_loss)))
                ep_ind += 1
            self.writer.add_scalar('epoch_loss', np.mean(recent_losses), epoch)
        time_end = time.time()
        print("fit took %s seconds" % str(time_end - time_start))
        print("feed_total: %s" % str(feed_total))
        print("run_total: %s" % str(run_total))

    def add_training_loss(self):
        """Adds training loss and scores for network."""
        pred = self.get_scores()
        loss = tf.losses.mean_squared_error(
            labels=self.test_label_placeholder, predictions=pred, weights=self.test_weight_placeholder)
        return pred, loss

    def get_scores(self):
        """Adds tensor operations for computing scores.

        Computes prediction yhat (eqn (1) in Matching networks) of class for test
        compounds.
        """
        # Get featurization for test
        # Shape (n_test, n_feat)
        test_feat = self.model.get_test_output()
        # Get featurization for support
        # Shape (n_support, n_feat)
        support_feat = self.model.get_support_output()

        # Computes the inner part c() of the kernel
        # (the inset equation in section 2.1.1 of Matching networks paper).
        # Normalize
        g = model_ops.cosine_distances(test_feat, support_feat)
        # Note that gram matrix g has shape (n_test, n_support)

        # soft corresponds to a(xhat, x_i) in eqn (1) of Matching Networks paper
        # https://arxiv.org/pdf/1606.04080v1.pdf
        # Computes softmax across axis 1, (so sums distances to support set for
        # each test entry) to get attention vector
        # Shape (n_test, n_support)
        attention = tf.nn.softmax(g)  # Renormalize
        # Weighted sum of support labels
        # Shape (n_support, 1)
        support_labels = tf.expand_dims(self.support_label_placeholder, 1)
        # pred is yhat in eqn (1) of Matching Networks.
        # Shape squeeze((n_test, n_support) * (n_support, 1)) = (n_test,1)
        pred = tf.matmul(attention, support_labels)
        return pred

    def predict(self, support, test):
        """Makes predictions on test given support.
        """
        return self.predict_proba(supprt, test)

    def predict_proba(self, support, test):
        """Makes predictions on test given support.

        Parameters
        ----------
        support: dc.data.Dataset
            The support dataset
        test: dc.data.Dataset
            The test dataset
        """
        y_preds = []
        for (X_batch, y_batch, w_batch, ids_batch) in test.iterbatches(
                self.test_size, deterministic=True):
            test_batch = NumpyDataset(X_batch, y_batch, w_batch, ids_batch)
            y_pred_batch = self.predict_proba_on_batch(support, test_batch)
            y_preds.append(y_pred_batch)
        y_pred = np.concatenate(y_preds)
        return y_pred

    def predict_on_batch(self, support, test_batch):
        """Make predictions on batch of data."""
        return self.predict_proba_on_batch(support, test_batch)

    def predict_proba_on_batch(self, support, test_batch):
        """Make predictions on batch of data."""
        n_samples = len(test_batch)
        X, y, w, ids = pad_batch(self.test_size, test_batch.X, test_batch.y,
                                 test_batch.w, test_batch.ids)
        padded_test_batch = NumpyDataset(X, y, w, ids)
        feed_dict = self.construct_feed_dict(padded_test_batch, support)
        # Get scores
        pred = self.sess.run(self.pred_op, feed_dict=feed_dict)
        # Remove padded elements
        pred = np.asarray(pred[:n_samples])
        return pred

    def evaluate(self, dataset, metric_dict, n_trials=100, log_every=50):
        """Evaluate performance on dataset according to metrics


        Evaluates the performance of the trained model by sampling supports randomly
        for each task in dataset. For each sampled support, the accuracy of the
        model with support provided is computed on all data for that task, then averaged
        for the number of trials run.
        Note that the support set is excluded from the accuracy calculation. 

        Parameters
        ----------
        dataset: dc.data.Dataset
            Dataset to test on.
        metrics: dc.metrics.Metric
            Evaluation metric.

        n_trials: int, optional
            Number of time, predict should be performed
        """
        # Get batches
        test_tasks = range(len(dataset.get_task_list()))
        task_scores = dict((x, ddict(list)) for x in metric_dict.keys())
        task2name = {}
        support_generator = SupportGenerator(
            dataset, self.support_size, n_trials)
        for ind, (taskind, task, support) in enumerate(support_generator):
            if ind % log_every == 0:
                print("Eval sample %d from task %s" % (ind, str(task)))
            # multitask case with missing data...
            task_dataset = dataset.get_task_dataset_minus_support(
                support, taskind)
            y_pred = self.predict_proba(support, task_dataset)
            task2name[taskind] = task
            for metkey, metric in metric_dict.items():
                task_scores[metkey][task].append(
                    metric.compute_metric(task_dataset.y, y_pred, task_dataset.w))

        # Join information for all tasks.
        mean_task_scores = ddict(dict)
        std_task_scores = ddict(dict)
        for metkey in metric_dict.keys():
            for taskind in test_tasks:
                task = task2name[taskind]
                mean_task_scores[metkey][task] = np.mean(
                    np.array(task_scores[metkey][task]))
                std_task_scores[metkey][task] = np.std(
                    np.array(task_scores[metkey][task]))
        return mean_task_scores, std_task_scores
