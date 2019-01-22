# -*- coding: utf-8 -*-
import sys
import pickle
import warnings
import pandas as pd
import numpy as np
import torch, os, argparse
from collections import defaultdict
from sklearn.model_selection import ParameterGrid, LeaveOneOut, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import explained_variance_score
from sklearn.utils import check_random_state
from metalearn.feature_extraction.transformers import FingerprintsTransformer
from pytoune.utils import torch_to_numpy
from torch.nn.functional import mse_loss
from .utils import to_unit

if not sys.warnoptions:
    warnings.simplefilter("ignore")

algos_classes = dict(kr=KernelRidge, gb=GradientBoostingRegressor, rf=RandomForestRegressor)

# Hyperparamters to explore for each algorithm
algos_grid = {'kr': [{"kernel": ['rbf'], "alpha": np.logspace(-3, 2, 6), "gamma": np.logspace(-3, 2, 6)},
                     {"kernel": ['linear'], "alpha": np.logspace(-3, 2, 6)}],
              'gb': {"n_estimators": 400},
              'rf': {"n_estimators": 400, 'n_jobs': -1}}


def transform_and_filter(x, y, fp):
    transformer = FingerprintsTransformer(kind=fp)
    try:
        x = transformer.transform(x)
    except Exception as e:
        x_filtered = []
        y_filtered = []
        for mol, y_val in zip(x, y):
            try:
                x_filtered.append(transformer.transform([mol])[0])
                y_filtered.append(y_val)
            except:
                print('here')
                pass
        x = np.array(x_filtered)
        y = np.array(y_filtered)
    return x, y


def fit_and_eval(episode, algo='rf', fp='morgan_circular'):
    x_train, y_train = transform_and_filter(*episode['Dtrain'], fp)
    x_test, y_test = transform_and_filter(*episode['Dtest'], fp)
    train_size = len(x_train)
    model_cls = algos_classes[algo]
    param_grid = algos_grid[algo]
    model = model_cls(**param_grid)
    if algo in ["gb", "rf"]:
        model = model_cls(**param_grid)
    else:
        model = GridSearchCV(model_cls(), param_grid, cv=train_size, refit=True, n_jobs=-1)
    model.fit(x_train, y_train.ravel())
    return y_test, model.predict(x_test)


class FPLearner:
    def __init__(self, algo, fp, *args, **kwargs):
        self.algo = algo
        self.fp = fp

    def fit(self, *args, **kwargs):
        pass

    def evaluate(self, metatest, metrics=[mse_loss], **kwargs):
        
        assert len(metrics) >= 1, "There should be at least one valid metric in the list of metrics "
        metrics_per_dataset = {metric.__name__: {} for metric in metrics}
        metrics_per_dataset["size"] = dict()
        for episodes in metatest:
            for (episode, _) in zip(*episodes):
                y_test, y_pred = fit_and_eval(episode, self.algo, self.fp)
                y_pred = torch.Tensor(y_pred.flatten())
                y_test = torch.Tensor(y_test.flatten())
                ep_idx = episode['idx']
                ep_name_is_new = (ep_idx not in metrics_per_dataset["size"])
                for metric in metrics:
                    m_value = to_unit(metric(y_pred, y_test))
                    if ep_name_is_new:
                        metrics_per_dataset[metric.__name__][ep_idx] = [m_value]
                    else:
                        metrics_per_dataset[metric.__name__][ep_idx].append(m_value)
                metrics_per_dataset['size'][ep_idx] = y_test.size(0)

        return metrics_per_dataset