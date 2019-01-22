import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class MetaDataset(Dataset):
    def __init__(self, x_transformer=None, y_transformer=None, task_descriptor_transformer=None, raw_inputs=False):
        self.x_transformer = x_transformer
        self.y_transformer = y_transformer
        self.task_descriptor_transformer = task_descriptor_transformer
        self.raw_inputs = raw_inputs

    def __iter__(self):
        raise NotImplementedError

    def _episode(self, xtrain, ytrain, xtest, ytest, task_descriptor=None, idx=None):
        if self.raw_inputs:
            return dict(Dtrain=(xtrain, ytrain),
                        Dtest=(xtest, ytest),
                        idx=idx,
                        tasks_descr=task_descriptor), ytest

        ntrain = len(xtrain)
        temp = self.x_transformer(np.concatenate([xtrain, xtest]))
        xtrain, xtest = temp[:ntrain], temp[ntrain:]
        temp = self.y_transformer(np.concatenate([ytrain, ytest]))
        ytrain, ytest = temp[:ntrain], temp[ntrain:]
        if task_descriptor is not None:
            task_descriptor = self.task_descriptor_transformer(task_descriptor)
            td = dict(task_descr = self.cuda_tensor(task_descriptor))
        else:
            td=dict()
        return (dict(Dtrain=(self.cuda_tensor(xtrain), self.cuda_tensor(ytrain)),
                     Dtest=(self.cuda_tensor(xtest), self.cuda_tensor(ytest)),
                     idx=idx,
                     **td),
                self.cuda_tensor(ytest))

    @staticmethod
    def cuda_tensor(x, use_available_gpu=True):
        if isinstance(x, torch.Tensor) and torch.cuda.is_available() and use_available_gpu:
            x = x.cuda()
        return x


class MetaRegressionDataset(MetaDataset):
    INF = int(1e5)

    def __init__(self, tasks_filenames, *args,
                 max_examples_per_episode=None, max_test_examples=None, is_test=False, seed=42, **kwargs):
        super(MetaRegressionDataset, self).__init__(*args, **kwargs)
        self.tasks_filenames = tasks_filenames[:]
        self.max_examples_per_episode = max_examples_per_episode
        self.max_test_examples = 500 if max_test_examples is None else max_test_examples
        self.rgn = np.random.RandomState(seed)
        self.is_test = is_test

    def __len__(self):
        return (20 * len(self.tasks_filenames)) if self.is_test else self.INF

    def episode_loader(self, filename):
        raise NotImplementedError

    def task_sizes(self):
        return np.array([len(self.episode_loader(f)[0]) for f in self.tasks_filenames])

    def __getitem__(self, i):
        file_index = i % len(self.tasks_filenames)
        x, y, task_descriptor = self.episode_loader(self.tasks_filenames[file_index])
        n, indexes = len(x), np.arange(len(x))

        self.rgn.shuffle(indexes)
        if self.is_test:
            k = min(int(n/2), self.max_examples_per_episode)
            train_indexes, test_indexes = indexes[:k], indexes[k:k+self.max_test_examples]
        else:
            n = min(2*self.max_examples_per_episode, n)
            indexes = indexes[:n]
            k = int(n/2)
            train_indexes, test_indexes = indexes[:k], indexes[k:]
        y_train = y[train_indexes]
        std_zero = np.std(y[train_indexes]) == 0
        if std_zero:
            y_train = np.array(y_train) +  np.random.normal(0, np.mean(y_train)/10.0)
        # print(len(train_indexes), len(test_indexes), self.is_test)
        return self._episode(x[train_indexes], y_train,
                             x[test_indexes], y[test_indexes],
                             task_descriptor, file_index)


if __name__ == '__main__':
    print('Nothing to run here !')