import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def make_generator(x, y, batch_size=-1):
    if batch_size == -1:
        while True:
            yield x, y
    else:
        while True:
            for i in range(0, len(y), batch_size):
                yield x[i:i+batch_size], y[i:i+batch_size]


class MetaDataset:
    def __init__(self, x_transformer=None, y_transformer=None, task_descriptor_transformer=None, batch_size=1):
        self.x_transformer = x_transformer
        self.y_transformer = y_transformer
        self.task_descriptor_transformer = task_descriptor_transformer
        self.batch_size = batch_size
        self.use_available_gpu = True

    def train_test_split(self, test_size):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def _episode(self, xtrain, ytrain, xtest, ytest, task_descriptor=None, name=None):
        ntrain = len(xtrain)
        temp = self.x_transformer(np.concatenate([xtrain, xtest]))
        xtrain, xtest = temp[:ntrain], temp[ntrain:]
        temp = self.y_transformer(np.concatenate([ytrain, ytest]))
        ytrain, ytest = temp[:ntrain], temp[ntrain:]
        if task_descriptor is not None:
            task_descriptor = self.task_descriptor_transformer(task_descriptor)
            task_descriptor = self.cuda_tensor(task_descriptor)
        return (dict(Dtrain=(self.cuda_tensor(xtrain), self.cuda_tensor(ytrain)),
                     Dtest=(self.cuda_tensor(xtest), self.cuda_tensor(ytest)),
                     name=self.str_to_tensor(name, self.use_available_gpu),
                     task_descr=task_descriptor),
                self.cuda_tensor(ytest))

    @staticmethod
    def cuda_tensor(x, use_available_gpu=True):
        if torch.cuda.is_available() and use_available_gpu:
            x = x.cuda()
        return x

    @staticmethod
    def str_to_tensor(s, use_available_gpu=True):
        res = torch.IntTensor([ord(char) for char in s])
        return MetaDataset.cuda_tensor(res, use_available_gpu)


class MetaRegressionDataset(MetaDataset):
    TRAIN, TEST = 0, 1

    def __init__(self, tasks_filenames, x_transformer=None, y_transformer=None, task_descriptor_transformer=None,
                 max_examples_per_episode=None,
                 batch_size=1, mode=0, seed=42):
        super(MetaRegressionDataset, self).__init__(x_transformer, y_transformer, task_descriptor_transformer, batch_size)
        self.tasks_filenames = tasks_filenames[:]
        self.max_examples_per_episode = max_examples_per_episode
        self.mode = mode
        self.episodes_samples_indexes = []
        self.rgn = np.random.RandomState(seed)
        self.predefine_all_episodes_indexes()

    def predefine_all_episodes_indexes(self):
        self.episodes_samples_indexes = []
        ksize = self.max_examples_per_episode
        mm = 0
        for file_index, epfile in enumerate(self.tasks_filenames):
            y = self.episode_loader(epfile)[1]
            n_examples = len(y)
            all_examples_indexes_list = list(range(n_examples))
            all_examples_indexes_set = set(all_examples_indexes_list)
            self.rgn.shuffle(all_examples_indexes_list)
            if self.mode == self.TRAIN:
                for i in range(0, n_examples, ksize):
                    train_indexes = all_examples_indexes_list[i: i+ksize]
                    if len(train_indexes) == ksize and np.std(y[train_indexes]) > 0:
                        test_indexes = list(all_examples_indexes_set.difference(train_indexes))
                        self.rgn.shuffle(test_indexes)
                        test_indexes = test_indexes[:ksize]
                        self.episodes_samples_indexes.append((file_index, train_indexes, test_indexes))
            else:
                i = j = 0
                while i < 100:
                    self.rgn.shuffle(all_examples_indexes_list)
                    train_indexes = all_examples_indexes_list[:ksize]
                    if len(train_indexes) == ksize and np.std(y[train_indexes]) > 0:
                        j += 1
                        test_indexes = all_examples_indexes_list[ksize:ksize+500]
                        self.episodes_samples_indexes.append((file_index, train_indexes, test_indexes))
                    i += 1
            mm += n_examples
        self.rgn.shuffle(self.episodes_samples_indexes)

    def number_of_tasks(self):
        return len(self.tasks_filenames)

    def eval(self):
        """ This will change how the iter function works"""
        self.mode = self.TEST
        self.predefine_all_episodes_indexes()

    def train(self):
        """ This will change how the iter function works"""
        self.mode = self.TRAIN
        self.predefine_all_episodes_indexes()

    def train_test_split(self, test_size):
        if test_size < 1:
            test_size = int(len(self.tasks_filenames) * test_size)
        else:
            test_size = int(test_size)
        assert test_size < len(self.tasks_filenames), "The test size should be lower than the number of episodes"
        self.rgn.shuffle(self.tasks_filenames)
        metatest_episode_files = self.tasks_filenames[:test_size]
        metatrain_episode_files = self.tasks_filenames[test_size:]

        metatrain = self.__class__(metatrain_episode_files, self.x_transformer, self.y_transformer,
                                   self.task_descriptor_transformer,
                                   self.max_examples_per_episode, self.batch_size, self.mode)
        metatest = self.__class__(metatest_episode_files, self.x_transformer, self.y_transformer,
                                  self.task_descriptor_transformer,
                                  self.max_examples_per_episode, self.batch_size, self.mode)
        return metatrain, metatest

    def episode_loader(self, filename):
        raise NotImplementedError

    def __indexes2episode(self, episode_indexes):
        file_index, train_indexes, test_indexes = episode_indexes
        x, y, task_descriptor = self.episode_loader(self.tasks_filenames[file_index])
        return self._episode(x[train_indexes], y[train_indexes],
                             x[test_indexes], y[test_indexes],
                             task_descriptor, self.tasks_filenames[file_index])

    def __iter__(self):
        do_loop = True
        n = len(self.episodes_samples_indexes)
        while do_loop:
            for i in range(0, n, self.batch_size):
                x_batch, y_batch = zip(*[self.__indexes2episode(self.episodes_samples_indexes[j])
                                         for j in range(i, min(i+self.batch_size, n))])
                yield x_batch, y_batch
            do_loop = self.mode == self.TRAIN

    def __len__(self):
        return int(len(self.episodes_samples_indexes)/self.batch_size)

    def to_multitask_generator(self, train_size=None):
        raise NotImplementedError

    def train_test_split_for_multitask(self, test_size):
        raise NotImplementedError
    # def full_datapoints_generator(self):
    #     batch_size = self.batch_size * self.max_examples_per_episode
    #     xs, ys = zip(*[self.episode_loader(epfile) for epfile in self.episode_files])
    #     x_ = np.concatenate(xs, axis=0)
    #     y_ = np.concatenate(ys, axis=0)
    #     for x, y in make_generator(x_, y_, batch_size):
    #         yield MetaDataset.cuda_tensor(self.x_transformer(x)), MetaDataset.cuda_tensor(self.y_transformer(y))
    #
    # def full_datapoints(self):
    #     xs, ys = zip(*[self.episode_loader(epfile) for epfile in self.episode_files])
    #     x = np.concatenate(xs, axis=0)
    #     y = np.concatenate(ys, axis=0)
    #     return MetaDataset.cuda_tensor(self.x_transformer(x)), MetaDataset.cuda_tensor(self.y_transformer(y))
    #


if __name__ == '__main__':
    print('Nothing to run here !')