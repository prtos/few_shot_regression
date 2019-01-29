"""
Train low-data attn models on Tox21. Test on SIDER. Test last fold only.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import deepchem as dc
from metalearn.low_data.deepchem_meta import MetaGraphRegressor
from metalearn.low_data.data_loader import load_dataset
from sklearn.metrics import r2_score
import pickle
import os

feps = 1e-7

def vse_metric(y_true, y_pred, w=None):
    errors = np.std((y_pred - y_true)**2)
    return errors

def pcc_metric(y_true, y_pred, w=None):
    mean_y_pred = np.mean(y_pred)
    mean_y_true = np.mean(y_true)
    y_predm = y_pred - mean_y_pred
    y_truem = y_true - mean_y_true
    r_num = np.sum(y_predm * y_truem)
    r_den = np.linalg.norm(y_predm, 2) * np.linalg.norm(y_truem, 2)
    r = r_num / (r_den + feps)
    r = np.clip(r, -1.0, 1.0)
    return r

def r2_metric(y_true, y_pred, w=None):
    return r2_score(y_true, y_pred)


def mse_metric(y_true, y_pred, w=None):
    return np.mean((y_pred - y_true) ** 2)

def get_model(name='lstm', test_size=10, train_size=10, max_depth=5, lr=1e-4, tboard='.logs/', n_feat=75):
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
    support_model.add_test(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    support_model.add_support(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    support_model.add(dc.nn.GraphPool())
    support_model.add(dc.nn.GraphConv(128, 64, activation='relu'))
    support_model.add_test(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    support_model.add_support(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    support_model.add(dc.nn.GraphPool())
    support_model.add(dc.nn.GraphConv(64, 128, activation='relu'))
    support_model.add_test(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    support_model.add_support(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    support_model.add(dc.nn.GraphPool())
    support_model.add(dc.nn.Dense(128, 64, activation='tanh'))
    support_model.add_test(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
    support_model.add_support(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
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

def run_model(params, output, data_path):
    dtname = params.get('dataset', 'metaqsar')
    nb_epochs = int(params.get('epoch', 10))
    n_train_trials = int(params.get('train_episodes', 100))
    n_eval_trials = int(params.get('test_episodes', 100))
    log_every = int(params.get('log_every', 100))
    test_size = int(params.get('test_size', 5))
    train_size = int(params.get('train_size', 5))
    name = params.get('name', 'lstm')
    tboard = os.path.join(output, "logs_{}/".format(name))
    max_tasks = int(params.get('max_tasks', 0))
    min_size = int(params.get('min_size', 0))
    if not max_tasks:
        max_tasks = None
    model = get_model(test_size=test_size, train_size=train_size, tboard=tboard)
    metrics = dict(
        r2=r2_metric,
        pcc=pcc_metric,
        mse=mse_metric,
        vse=vse_metric
    )

    train, test = load_dataset(dtname, max_tasks=max_tasks, min_size=min_size)
    model.fit(train,
              nb_epochs=nb_epochs,
              n_episodes_per_epoch=n_train_trials,
              log_every=log_every)
    # this contains the mean score per task using n_eval_trials
    mean_scores, std_scores = model.evaluate(
        test, metrics, n_trials=n_eval_trials)


    with open(os.path.join(output, "score_{}_{}_{}_{}.pkl".format(name, dtname, train_size, test_size)), 'wb') as out_file:
        pickle.dump({'score':mean_scores, 'std':std_scores}, out_file, protocol=4)

    for name in metrics.keys():
        print('==> Using metric {}'.format(name))
        # Train support model on train
        print("\tMean Scores on evaluation dataset")
        print("\t", mean_scores[name])
        print("\tStandard Deviations on evaluation dataset")
        print("\t", std_scores[name])


if __name__ == '__main__':
    run_model({'epoch': 1, 'max_tasks':10, 'train_episodes': 10, 'test_episodes':5, 'dataset':'pubchem', 'train_size':5, 'test_size':5}, './', None)