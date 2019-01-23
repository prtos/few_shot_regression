"""
Load datasets for Low Data processing.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import time
import numpy as np
import deepchem as dc
import logging
import pandas as pd
from deepchem.data import NumpyDataset
from os.path import dirname, realpath, join
from os import listdir
from sklearn.model_selection import train_test_split
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops
from rdkit import Chem

logger = logging.getLogger(__name__)
DATASETS_ROOT = join(dirname(dirname(dirname(realpath(__file__)))), 'datasets')

def count_lines(filename, offset):
    nlines = sum(1 for i in open(filename, 'rb') if i.strip())
    return nlines - offset

def to_numpy_dataset(X, y, w=None, ids=None):
    """Converts dataset to numpy dataset."""
    return dc.data.NumpyDataset(X, y, w, ids)


def featurize_smiles(arr):
    featurizer = dc.feat.ConvMolFeaturizer()
    features = []
    for ind, elem in enumerate(arr.tolist()):
        mol = Chem.MolFromSmiles(elem)
        if mol:
            new_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, new_order)
        features.append(featurizer([mol]))

    valid_inds = np.array(
        [1 if elt.size > 0 else 0 for elt in features], dtype=bool)
    features = [elt for (is_valid, elt) in zip(
        valid_inds, features) if is_valid]
    features = np.squeeze(np.array(features))
    return features.reshape(-1,), valid_inds


class MetaRegressionDataset:
    def __init__(self, tasks_filenames, x_transformer=None, y_transformer=None, seed=42):
        super(MetaRegressionDataset, self).__init__()
        self.tasks_filenames = tasks_filenames[:]
        self.x_transformer = x_transformer
        self.y_transformer = y_transformer
        self.rgn = np.random.RandomState(seed)
        self.file2tasks = {}
        self.dataset = {}

    def get_sampling_weights(self):
        episode_sizes = np.array([len(self.get_task(t))
                                  for t in self.tasks_filenames])
        episode_sizes = np.log2(episode_sizes)
        return episode_sizes/np.sum(episode_sizes)

    def __len__(self):
        return len(self.tasks_filenames)

    def get_task(self, task):
        if isinstance(task, int):
            task = self.tasks_filenames[task]
        taskname = self.file2tasks[task]
        return self.dataset[taskname]

    def __getitem__(self, ind):
        raise NotImplemented

    def get_single_task_train(self, n_sample, n_test, task):
        return self.get_task_train(1, n_sample, n_test, task)[0]

    def get_task_support(self, n_sample, task):
        """Generates one support set purely for specified task.
        """
        dataset = self.__getitem__(task)
        max_sample = dataset.y.shape[0]
        # No replacement allowed for supports
        ids = self.rgn.choice(max_sample, (n_sample,), replace=False)
        # Handle one-d vs. non one-d feature matrices
        X = dataset.X[ids]
        y = np.expand_dims(dataset.y[ids, :], 1)
        w = np.expand_dims(dataset.w[ids, :], 1)
        return self.file2tasks[self.tasks_filenames[task]], NumpyDataset(X, y, w, ids)

    def get_task_dataset_minus_support(self, support, task):
        """Gets data for specified task, minus support points.

        Useful for evaluating model performance once trained (so that
        test compounds can be ensured distinct from support.)

        Parameters
        ----------
        dataset: dc.data.Dataset
            Source dataset.
        support: dc.data.Dataset
            The support dataset
        task: int
            Task number of task to select.
        """
        dataset = self.__getitem__(task)
        support_ids = set(support.ids)
        non_support_inds = [
            ind for ind in range(len(dataset)) if dataset.ids[ind] not in support_ids
        ]

        # Remove support indices
        X = dataset.X[non_support_inds]
        y = dataset.y[non_support_inds, :]
        w = dataset.w[non_support_inds, :]
        ids = dataset.ids[non_support_inds]
        return NumpyDataset(X, y, w, ids)

    def get_task_train(self, n_episodes, n_sample, n_test, task):
        """Generates one support set purely for specified task.
        """
        time_start = time.time()
        dataset = self.__getitem__(task)
        max_sample = dataset.y.shape[0]
        supports = []
        tests = []
        for episode in range(n_episodes):
            # No replacement allowed for supports
            ids = self.rgn.choice(
                max_sample, (n_sample+n_test,), replace=False)
            # Handle one-d vs. non one-d feature matrices
            train_indexes, test_indexes = ids[:n_sample], ids[n_sample:]
            X = dataset.X[train_indexes]
            y = np.expand_dims(dataset.y[train_indexes, :], 1)
            w = np.expand_dims(dataset.w[train_indexes, :], 1)
            supports.append(NumpyDataset(X, y, w, train_indexes))
            X = dataset.X[test_indexes]
            y = np.expand_dims(dataset.y[test_indexes, :], 1)
            w = np.expand_dims(dataset.w[test_indexes, :], 1)
            tests.append(NumpyDataset(X, y, w, test_indexes))
        return list(zip(supports, tests))


class MetaQSARdatatset(MetaRegressionDataset):
    def __init__(self, *args, **kwargs):
        super(MetaQSARdatatset, self).__init__(*args, **kwargs)
        self.x_transformer = featurize_smiles
        self.y_transformer = lambda y: y

    def __getitem__(self, ind):
        filename = self.tasks_filenames[ind % len(self.tasks_filenames)]
        taskname = self.file2tasks.get(filename)
        if taskname:
            return self.get_task(filename)
        with open(filename, 'r') as f_in:
            protein = f_in.readline().split()[0].upper()
            # measurement = f_in.readline()
        data = pd.read_csv(filename, header=None, skiprows=2,
                           delim_whitespace=True).values
        #print(data[:10, 0])
        x, y = data[:, 1], data[:, 2].astype('float').reshape((-1, 1))
        x, inds = self.x_transformer(x)
        y = self.y_transformer(y)
        y = y[inds, :]
        self.dataset[protein] = to_numpy_dataset(x, y)
        self.file2tasks[filename] = protein
        return self.dataset[protein]

    def get_task_list(self):
        return self.tasks_filenames


class PubChemdatatset(MetaQSARdatatset):

    def __getitem__(self, ind):
        filename = self.tasks_filenames[ind % len(self.tasks_filenames)]
        taskname = self.file2tasks.get(filename)
        if taskname:
            return self.get_task(filename)
        with open(filename, 'r') as f_in:
            protein = f_in.readline().split()[0].upper()
            # measurement = f_in.readline()
        data = pd.read_csv(filename, header=None, skiprows=1).values
        #print(data[:10, 0])
        x, y = data[:, 0], data[:,1].astype('float').reshape((-1, 1))
        x, inds = self.x_transformer(x)
        y = self.y_transformer(y)
        y = y[inds, :]
        self.dataset[protein] = to_numpy_dataset(x, y)
        self.file2tasks[filename] = protein
        return self.dataset[protein]


def __get_partitions(episode_files, test_size, **kwargs):
    train_files, test_files = train_test_split(
        episode_files, test_size=test_size)
    # get, train, valid, test data
    return train_files, test_files


def load_dataset(dataset_name, ds_folder=None, max_tasks=None, test_size=0.25, min_size=10, **kwargs):
    maps = dict(
        metaqsar=('chembl', '.tsv', MetaQSARdatatset),
        pubchem=('pubchemtox', '.csv', PubChemdatatset)

    )
    if dataset_name not in maps:
        raise Exception(f"Unhandled dataset. The name of \
            the dataset should be one of those: {list(maps.keys())}")
    folder, ext, dscls = maps[dataset_name]
    ds_folder = join((ds_folder or DATASETS_ROOT), folder)
    jfile = os.path.join(ds_folder, folder+".json")
    if os.path.exists(jfile):
        dt = json.load(open(jfile)) 
        train_files =  dt['Dtrain']
        test_files = dt['Dtest']
    else:
        files = [join(ds_folder, x) for x in listdir(ds_folder)
             if x.endswith(ext)][:max_tasks]
        train_files, test_files = __get_partitions(files, test_size=test_size, **kwargs)
    if min_size: # I am allowed to do this since the number of sample with less than 20 
        train_files = [f for f in train_files if count_lines(f)> min_size]
        test_files = [f for f in test_files if count_lines(f)> min_size]
    train = dscls(train_files, **kwargs)
    test = dscls(test_files, **kwargs)
    return train, test


class EpisodeGenerator(object):
    """Generates (support, test) pairs for episodic training.

    Precomputes all (support, test) pairs at construction. Allows to reduce
    overhead from computation.
    """

    def __init__(self, dataset, n_train, n_test, n_episodes_per_task):
        """
        Parameters
        ----------
        dataset: dc.data.Dataset
            Holds dataset from which support sets will be sampled.
        n_train: int
            Number of samples in support set.
        n_test: int
            Number of samples in test set.
        n_episodes_per_task: int
            Number of (support, task) pairs to sample in total.
        replace: bool
            Whether to use sampling with or without replacement.
        """
        time_start = time.time()
        self.tasks = np.arange(len(dataset.get_task_list()))
        self.n_tasks = len(self.tasks)
        self.n_episodes_per_task = n_episodes_per_task
        self.n_train, self.n_test = n_train, n_test
        self.dataset = dataset
        self.task_episodes = {}
        self.rng = dataset.rgn

        for task in self.tasks:
            task_data = self.dataset.get_task_train(1, n_train, n_test, task)
            self.task_episodes[task] = task_data

        self.task_weights = self.dataset.get_sampling_weights()
        # Set initial iterator state
        self.trial_num = 0
        time_end = time.time()
        logger.info("Constructing EpisodeGenerator took %s seconds" %
                    str(time_end - time_start))

    def __iter__(self):
        return self

    def next(self):
        """Sample next (support, test) pair.

        Return from internal storage.
        """
        if self.trial_num == self.n_episodes_per_task:
            raise StopIteration
        else:
            # pick one task randomly
            task = self.rng.choice(self.tasks, 1, p=self.task_weights)[0]
            #support = self.supports[task][self.trial_num]
            support, test = self.dataset.get_single_task_train(
                self.n_train, self.n_test, task)
            self.trial_num += 1
            return (task, support, test)

    __next__ = next  # Python 3.X compatibility


class SupportGenerator(object):
    """Generate support sets from a dataset.

    Iterates over tasks and trials. For each trial, picks one support from
    each task, and returns in a randomized order
    """

    def __init__(self, dataset, n_sample, n_trials):
        """
        Parameters
        ----------
        dataset: dc.data.Dataset
            Holds dataset from which support sets will be sampled.
        n_sample: int
            Number samples.
        n_trials: int
            Number of passes per task to make. In total, n_tasks*n_trials
            support sets will be sampled by algorithm.
        """

        self.tasks = np.arange(len(dataset.get_task_list()))
        self.n_tasks = len(self.tasks)
        self.n_trials = n_trials
        self.dataset = dataset
        self.n_sample = n_sample

        # Init the iterator
        self.perm_tasks = np.random.permutation(
            self.tasks)  # avoid quick adaptation
        # Set initial iterator state
        self.task_num = 0
        self.trial_num = 0

    def __iter__(self):
        return self

    def next(self):
        """Sample next support.

        Supports are sampled from the tasks in a random order. Each support is
        drawn entirely from within one task.
        """
        if self.trial_num == self.n_trials:
            raise StopIteration
        else:
            task = self.perm_tasks[self.task_num]  # Get id from permutation
            #support = self.supports[task][self.trial_num]
            taskname, support = self.dataset.get_task_support(
                self.n_sample, task)
            # Increment and update logic
            self.task_num += 1
            if self.task_num == self.n_tasks:
                self.task_num = 0  # Reset
                self.perm_tasks = np.random.permutation(
                    self.tasks)  # Permute again
                self.trial_num += 1  # Upgrade trial index

            return (task, taskname, support)

    __next__ = next  # Python 3.X compatibility
