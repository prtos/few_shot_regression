import json
import torch
import numpy as np
import pandas as pd
from functools import partial
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
                              RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.base import BaseEstimator, clone
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import MultiTaskElasticNet, MultiTaskLasso
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from joblib import dump, load, Parallel, delayed
from modAL.models import ActiveLearner
from modAL.batch import ranked_batch
from metalearn.models.factory import ModelFactory
from metalearn.feature_extraction import SequenceTransformer, AdjGraphTransformer, FingerprintsTransformer
from metalearn.feature_extraction.constants import SMILES_ALPHABET
from metalearn.scaler import get_scaler
from metalearn.metric import std_se, mse, mae, r2_score, pearsonr, accuracy, f1_score, roc_auc_score

SUPPORTED_FINGERPRINTS = ['meta', 'atom', 'estate', 'maccs', 'morgan_circular', 'topo', 'properties', 'rdkit']


def unflatten(dictionary):
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict


def predict_fn(x, mod):  # transformer, x_scaler, model):
    #.to_fp, self.x_scaler, self.model)
    x_, valid_idx = mod.to_fp(x)
    x_ = mod.x_scaler.transform(x_)
    y = mod.model.predict(x_)
    return y, np.array(valid_idx)


def compute_metrics(metrics, y_test, y_pred):
    metrics_names = [f.__name__ for f in metrics]
    res = dict()
    for metric_name, metric in zip(metrics_names, metrics):
        res[metric_name] = metric(y_test, y_pred)
    return res


def gp_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    return std


def rf_regression_std(regressor, X):
    temp = np.hstack([estimator.predict(X).reshape(-1, 1) for estimator in regressor.estimator.estimators_])
    return np.std(temp, axis=1)


def uncertainty_batch_sampling(regressor, X, n_instances=10,
                               metric='euclidean', n_jobs=-1,
                               uncertainty_fn=rf_regression_std):
    uncertainty = rf_regression_std(regressor, X)
    query_indices = ranked_batch(regressor, unlabeled=X, uncertainty_scores=uncertainty,
                                 n_instances=n_instances, metric=metric, n_jobs=n_jobs)
    return query_indices, X[query_indices]


class MetaFPTransformer:
    is_loaded = False
    fp = None
    transformer = None

    @classmethod
    def load_meta_feature_extractor(cls, **kwargs):

        if not cls.is_loaded:
            param_file = kwargs.get('param_file', 'meta_model/params.json')
            model_file = kwargs.get('model_file', 'meta_model/model_ckp.ckp')
            with open(param_file) as fd:
                params = json.load(fd)

            params = unflatten(params)
            model_name, model_params = params['model_name'], params['model_params']
            model = ModelFactory()(model_name, **model_params)
            model.load(model_file)
            use_graph = params['dataset_params'].get('use_graph', False)
            if use_graph:
                cls.transformer = AdjGraphTransformer()
            else:
                cls.transformer = SequenceTransformer(SMILES_ALPHABET)
            cls.is_loaded = True
            cls.fp = model.model.feature_extractor
        return cls.fp

    def transform(self, x, *args):
        if not self.is_loaded:
            self.load_meta_feature_extractor()

        return self.fp(self.transformer.transform(x))

    def __call__(self, x, dtype=torch.long, cuda=False, **kwargs):
        if not self.is_loaded:
            self.load_meta_feature_extractor()
        temp = self.transformer.transform(x)
        fp = self.fp(temp).detach().numpy()

        # if is_dtype_numpy_array(dtype):
        #     fp = np.array(fp, dtype=dtype)
        # elif is_dtype_torch_tensor(dtype):
        #     fp = to_tensor(fp, gpu=cuda, dtype=dtype)
        # else:
        #     raise(TypeError('The type {} is not supported'.format(dtype)))
        return fp, np.arange(len(x))


class BaseModel(BaseEstimator):
    """Generic class for learning using sklearn models

    Parameters
    ----------
    name : str
        name of the model, should be one of the supported regression model type
    tag : str
        tag of the model for printing purpose
    params: list of dict or dict
        list of dict each one containing the parameter list for the choosen model
        Also accept a dict

    Attributes
    ----------
    name: name of the model
    model : float or torch.FloatTensor (1*1)
        L2-norm regularizer weight during the KRR learning phase
    params: dict
        Parameter dict for the base model inside the Estimator
    gridcv: bool
        whether to run grid search or not
    """
    SUPPORTED_ESTIMATOR = {
        'krr_rgr': KernelRidge(),
        'rf_rgr': RandomForestRegressor(),
        'enet_rgr': MultiTaskElasticNet(),
        'ab_rgr': AdaBoostRegressor(),  # need multioutput
        'gb_rgr': GradientBoostingRegressor(),  # need multioutput
        'lasso_rgr': MultiTaskLasso(),
        'rf_clf': RandomForestClassifier(class_weight='balanced', n_jobs=-1),  # inheritently multioutput
        'gb_clf': GradientBoostingClassifier(),  # need multioutput
        'ets_clf': ExtraTreesClassifier(class_weight='balanced'),  # inheritently multioutput
        'kn_clf': KNeighborsClassifier(),  # inheritently multioutput
        'rn_clf': RadiusNeighborsClassifier(),  # inheritently multioutput
        'dt_clf': DecisionTreeClassifier(class_weight='balanced'),  # inheritently multioutput
        'et_clf': ExtraTreeClassifier(class_weight='balanced'),  # inheritently multioutput
    }
    METRICS_RGR = [std_se, mse, mae, r2_score, pearsonr]
    METRICS_CLF = [accuracy, f1_score, roc_auc_score]

    def __init__(self, model_name, fp_name, tag=None, multitask=False):
        super(BaseModel, self).__init__()
        if not model_name or model_name not in self.SUPPORTED_ESTIMATOR:
            raise ValueError("Algorithm {} not supported !".format(model_name))
        if not fp_name or fp_name not in SUPPORTED_FINGERPRINTS:
            raise ValueError("Fingerprint {} not supported !".format(fp_name))
        self.model_name = model_name
        self.tag = tag
        self.fp_name = fp_name
        self.x_scaler = None
        self.y_scaler = None
        self.multitask = multitask
        if model_name.endswith('_rgr'):
            self.metrics = self.METRICS_RGR
        elif model_name.endswith('_clf'):
            self.metrics = self.METRICS_CLF
        else:
            raise Exception('')
        if fp_name == 'meta':
            MetaFPTransformer.load_meta_feature_extractor()
        # self.fp_feat = (MetaFPTransformer() if self.fp_name == 'meta'
        #                 else FingerprintsTransformer(self.fp_name))

    def init_model(self, name):
        self.model = self.SUPPORTED_ESTIMATOR[name]
        if self.model_name in ["gb_rgr", "ab_rgr"] and self.multitask:
            self.model = MultiOutputRegressor(self.model)
        if self.model_name in ["gb_clf", "ab_clf"] and self.multitask:
            self.model = MultiOutputClassifier(self.model)

    def to_fp(self, x):
        fp_feat = (MetaFPTransformer() if self.fp_name == 'meta'
                   else FingerprintsTransformer(self.fp_name))
        res = fp_feat(x, dtype=np.float32, ignore_errors=True)
        return res

    def is_multi_tasks(self):
        return len(self.tasks) > 1

    def fit(self, X, y, param_grid=None, x_scaler='none', y_scaler='none', cv=5, n_jobs=-1, verbose=0, **kwargs):
        self.init_model(self.model_name)
        print('fitting.....')
        self.x_scaler = get_scaler(x_scaler)
        self.y_scaler = get_scaler(y_scaler)
        x_, valid_idx = self.to_fp(X)
        x_ = self.x_scaler.fit_transform(x_)
        y_ = self.y_scaler.fit_transform(y[valid_idx])
        if isinstance(cv, float):
            train_idx, test_idx = train_test_split(np.range(len(x_)), cv)
            cv = ((train_idx, test_idx) for _ in range(1))
        if param_grid is not None and len(ParameterGrid(param_grid)) > 1:
            trainer = GridSearchCV(self.model, param_grid, cv=cv, refit=True, n_jobs=n_jobs, verbose=verbose)
            trainer.fit(x_, y_)
            self.model = trainer.best_estimator_
        else:
            if param_grid is None:
                params = dict()
            else:
                params_list = list(ParameterGrid(param_grid))
                params = dict() if len(params_list) == 0 else params_list[0]
            self.model = self.model.set_params(**params)
            self.model.fit(x_, y_)
        return self

    def predict(self, X, batch_size=128, n_jobs=-1, verbose=0, **kwargs):
        print('predicting.....')
        n = len(X)
        res = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(predict_fn)(X[i:i + batch_size], self)
            for i in range(0, n, batch_size))
        ys, vs = zip(*res)
        y = np.concatenate(ys, axis=0)
        valid_idx = np.concatenate([deb + vs[i] for i, deb in enumerate(range(0, n, batch_size))], axis=0)
        return y, valid_idx

    def eval(self, X, y, **kwargs):
        y_pred, valid_idx = self.predict(X, **kwargs)
        y_test = self.y_scaler.transform(y[valid_idx])
        y_pred = y_pred.reshape(y_test.shape)
        return compute_metrics(self.metrics, y_test, y_pred)

    def fit_eval(self, X, y, train_size=None, test_size=None, repeats=1, random_state=None, **kwargs):
        assert (train_size is not None) or (test_size is not None)
        rng = np.random.RandomState(random_state)
        c_res = []
        for _ in range(repeats):
            x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=train_size,
                                                                test_size=test_size, random_state=rng, shuffle=True)
            self.fit(x_train, y_train, **kwargs)
            res = self.eval(x_test, y_test, **kwargs)
            res.update(dict(n_eval=len(y_test), n_train=len(y_train)))
            c_res.append(res)

        means = pd.DataFrame(c_res).apply(np.mean, axis=0).to_dict()
        stds = pd.DataFrame(c_res).apply(np.std, axis=0).to_dict()
        res = {k + '_mean': v for k, v in means.items()}
        res.update({k + '_std': v for k, v in stds.items()})
        return res

    def save_model(self, filename):
        dump(self, filename)

    @classmethod
    def load_model(cls, filename):
        return load(filename)


class ActiveModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super(ActiveModel, self).__init__(*args, **kwargs)

    def fit(self, X, y, initial_train_size, max_queries, n_queries_per_batch=10, param_grid=None, x_scaler='none',
            y_scaler='none', cv=5, n_jobs=-1, verbose=0, **kwargs):
        self.init_model(self.model_name)
        if self.model_name == 'rf_rgr':
            strategy = partial(uncertainty_batch_sampling, n_jobs=n_jobs,
                               n_instances=n_queries_per_batch,
                               uncertainty_fn=rf_regression_std)
        elif self.model_name == 'gp_rgr':
            strategy = partial(uncertainty_batch_sampling, n_jobs=n_jobs,
                               n_instances=n_queries_per_batch,
                               uncertainty_fn=gp_regression_std)
        else:
            raise Exception("No strategy in place for the specified model name")

        print('fitting.....')
        self.x_scaler = get_scaler(x_scaler)
        self.y_scaler = get_scaler(y_scaler)
        x_, valid_idx = self.to_fp(X)
        x_train, x_pool, y_train, y_pool = train_test_split(x_, y[valid_idx], train_size=initial_train_size)
        x_train = self.x_scaler.fit_transform(x_train)
        y_train = self.y_scaler.fit_transform(y_train)
        x_pool = self.x_scaler.transform(x_pool)
        y_pool = self.y_scaler.transform(y_pool)
        if isinstance(cv, float):
            train_idx, test_idx = train_test_split(np.range(len(x_train)), cv)
            cv = ((train_idx, test_idx) for _ in range(1))
        if param_grid is not None and len(ParameterGrid(param_grid)) > 1:
            trainer = GridSearchCV(self.model, param_grid, cv=cv, refit=True, n_jobs=n_jobs, verbose=verbose)
            trainer.fit(x_train, y_train)
            base_model = trainer.best_estimator_
        else:
            if param_grid is None:
                params = dict()
            else:
                params_list = list(ParameterGrid(param_grid))
                params = dict() if len(params_list) == 0 else params_list[0]
            base_model = self.model.set_params(**params)
            base_model.fit(x_train, y_train)

        active_model = ActiveLearner(
            estimator=base_model,
            query_strategy=strategy,
            X_training=x_train, y_training=(y_train.reshape(-1, 1) if len(y_train.shape) == 1 else y_train)
        )

        best_score, last_decrease = None, 0
        for i in range(max_queries):
            query_idx, query_instances = active_model.query(x_pool)
            active_model.teach(x_pool[query_idx], y_pool[query_idx])
            # Remove the queried instance from the unlabeled pool.
            x_pool, y_pool = np.delete(x_pool, query_idx, axis=0), np.delete(y_pool, query_idx, axis=0)
            preds = active_model.predict(x_pool).reshape(y_pool.shape)
            print(compute_metrics(self.metrics, y_pool, preds))
            if self.model_name.endswith('_rgr'):
                score = mse(y_pool, preds)
            else:
                score = 1 - accuracy(y_pool, preds)
            if best_score is None or score < best_score:
                best_score = score
                last_decrease = 0
            else:
                last_decrease += 1

            if last_decrease >= 10:
                break
            if len(x_pool) == 0:
                break

        self.model = active_model.estimator
        return self
