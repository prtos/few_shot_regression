"""
Train low-data attn models on Tox21. Test on SIDER. Test last fold only.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import deepchem as dc
import tensorflow as tf
from deepchem_meta import MetaGraphRegressor
from data_loader import load_dataset


def get_model(name='lstm', test_size=10, train_size=10, max_depth=3, lr=1e-3, tboard='.logs/', n_feat=75):
    """
        Builds an return a regressor model, extending the work of 
        See https://arxiv.org/pdf/1606.04080v1.pdf.
        Note that there is no meta-validation for these mode

        Parameters
        ----------
        name: str
            one of 'lstm' for iterative lstm or 'siamese' (see paper)
        test_size: int
            number of samples in test of meta-train
        train_size: int
            Number of samples in train of meta-train
        max_depth: int
            Depth of the iterLSTM if applicable
        lr: float
            Learning rate
        tboard: str
            Path to tensorboard X saving folder
    """
    support_model = dc.nn.SequentialSupportGraph(n_feat)
    # Add layers
    support_model.add(dc.nn.GraphConv(64, n_feat, activation='relu'))
    support_model.add(dc.nn.GraphPool())
    support_model.add(dc.nn.GraphConv(128, 64, activation='relu'))
    support_model.add(dc.nn.GraphPool())
    support_model.add(dc.nn.GraphConv(64, 128, activation='relu'))
    support_model.add(dc.nn.GraphPool())
    support_model.add(dc.nn.Dense(128, 64, activation='tanh'))

    support_model.add_test(dc.nn.GraphGather(test_size, activation='tanh'))
    support_model.add_support(dc.nn.GraphGather(train_size, activation='tanh'))
    if name.lower() != 'siamese':
        support_model.join(dc.nn.ResiLSTMEmbedding(
            test_size, train_size, 128, max_depth))
    model = MetaGraphRegressor(
        support_model,
        test_size=test_size,
        support_size=train_size,
        learning_rate=lr)
    return model


if __name__ == '__main__':
    nb_epochs = 50
    n_train_trials = 100
    n_eval_trials = 10
    log_every = 30
    max_tasks = 100
    model = get_model('siamese', test_size=5, train_size=5, tboard="./logs/siamese")
    metrics = dict(
        r2=dc.metrics.Metric(dc.metrics.pearson_r2_score,
                             mode="regression", verbose=False),
        rms=dc.metrics.Metric(dc.metrics.rms_score,
                              mode="regression", verbose=False)
    )
    np.random.seed(42)
    train, test = load_dataset('metaqsar', max_tasks=max_tasks)
    model.fit(train,
              nb_epochs=nb_epochs,
              n_episodes_per_epoch=n_train_trials,
              log_every=log_every)
    # this contains the mean score per task using n_eval_trials
    mean_scores, std_scores = model.evaluate(
        test, metrics, n_trials=n_eval_trials)
    for name in metrics.keys():
        print('==> Using metric {}'.format(name))
        # Train support model on train
        print("\tMean Scores on evaluation dataset")
        print("\t", mean_scores[name])
        print("\tStandard Deviations on evaluation dataset")
        print("\t", std_scores[name])
