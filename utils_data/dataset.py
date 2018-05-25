import os
import numpy as np
import torch
from keras.utils.np_utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split


def load_images_data(grouped_filenames):
    x = np.array([Image.open(filename) for class_filenames in grouped_filenames for filename in class_filenames])
    y = np.array([i for i, class_filenames in enumerate(grouped_filenames) for _ in class_filenames])
    y = to_categorical(y, len(grouped_filenames))
    return x, y


def make_generator(x, y, batch_size=-1):
    if batch_size == -1:
        while True:
            yield x, y
    else:
        while True:
            for i in range(0, len(y), batch_size):
                yield x[i:i+batch_size], y[i:i+batch_size]


class MetaDataset:
    def __init__(self, x_transformer=None, y_transformer=None, batch_size=1):
        self.x_transformer = x_transformer
        self.y_transformer = y_transformer
        self.batch_size = batch_size
        self.use_available_gpu = True

    def train_test_split(self, test_size):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def _episode(self, xtrain, ytrain, xtest, ytest, name=None):
        if len(ytest) > 5000:
            xtest, ytest = xtest[:5000], ytest[:5000]
        fx, fy = self.x_transformer, self.y_transformer
        xtrain, ytrain = fx(xtrain), fy(ytrain)
        xtest, ytest = fx(xtest), fy(ytest)
        return (dict(Dtrain=(self.cuda_tensor(xtrain), self.cuda_tensor(ytrain)),
                     Dtest=self.cuda_tensor(xtest),
                     name=self.str_to_tensor(name, self.use_available_gpu)),
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

    def __init__(self, episode_files, x_transformer=None, y_transformer=None, max_examples_per_episode=None,
                 batch_size=1, train_without_dtest=False):
        super(MetaRegressionDataset, self).__init__(x_transformer, y_transformer, batch_size)
        self.episode_files = episode_files[:]
        self.max_examples_per_episode = max_examples_per_episode
        self.mode = self.TRAIN
        self.train_without_dtest = train_without_dtest

    def number_of_tasks(self):
        return len(self.episode_files)

    def eval(self):
        """ This will change how the iter function works"""
        self.mode = self.TEST

    def train(self):
        """ This will change how the iter function works"""
        self.mode = self.TRAIN

    def train_test_split(self, test_size):
        if test_size < 1:
            test_size = int(len(self.episode_files) * test_size)
        else:
            test_size = int(test_size)
        assert test_size < len(self.episode_files), "The test size should be lower than the number of episodes"

        np.random.shuffle(self.episode_files)
        metatest_episode_files = self.episode_files[:test_size]
        metatrain_episode_files = self.episode_files[test_size:]

        metatrain = self.__class__(metatrain_episode_files, self.x_transformer, self.y_transformer,
                                   self.max_examples_per_episode, self.batch_size)
        metatest = self.__class__(metatest_episode_files, self.x_transformer, self.y_transformer,
                                  self.max_examples_per_episode, self.batch_size)
        return metatrain, metatest

    def episode_loader(self, filename):
        raise NotImplementedError

    def get_sampling_weights(self):
        raise NotImplementedError

    def __filename2episode(self, episode_filename):
        # refactor this
        x, y = self.episode_loader(episode_filename)
        size_episode = len(y)
        idx = np.arange(size_episode)
        np.random.shuffle(idx)
        if ((self.max_examples_per_episode is None) or
                (size_episode <= self.max_examples_per_episode*2)):
            half_size = int(size_episode / 2)
            train_idx, test_idx = idx[:half_size], idx[half_size:]
        else:
            n = self.max_examples_per_episode
            train_idx, test_idx = idx[:n], idx[n:2*n]
            if self.mode == self.TEST:
                train_idx, test_idx = idx[:n], idx[n:]
        if self.train_without_dtest and (self.mode != self.TEST):
            return self._episode(x[train_idx], y[train_idx], x[train_idx], y[train_idx], episode_filename)
        return self._episode(x[train_idx], y[train_idx], x[test_idx], y[test_idx], episode_filename)

    def __iter__(self):
        N = len(self.episode_files)
        sampling_weights = self.get_sampling_weights()
        if self.mode == self.TEST:
            for i in range(0, 100*N, self.batch_size):
                episode_filenames = [self.episode_files[j%N] for j in range(i, i+self.batch_size)]
                yield [self.__filename2episode(epf) for epf in episode_filenames]
        else:
            while True:
                episode_filenames = np.random.choice(self.episode_files, p=sampling_weights, size=self.batch_size)
                x_batch, y_batch = zip(*[self.__filename2episode(epf) for epf in episode_filenames])
                yield x_batch, y_batch

    def to_multitask_generator(self, train_size=None):
        """

        :param train_size: positive value will select examples in the
        :return:
        """
        n = len(self.episode_files)
        sampling_weights = self.get_sampling_weights()
        episode_ids = np.arange(n)
        while True:
            episode_idx = np.random.choice(episode_ids, p=sampling_weights, size=self.batch_size)
            xs, ys, masks = [], [], []
            for i in episode_idx:
                x, y = self.episode_loader(self.episode_files[i])
                idx = np.arange(len(y))
                if train_size is not None:
                    idx_train, idx_test = train_test_split(idx, train_size=abs(train_size), shuffle=False)
                    idx = idx_train if train_size > 0 else idx_test
                np.random.shuffle(idx)
                train_idx = idx[:self.max_examples_per_episode]
                x_, y_ = x[train_idx], y[train_idx]
                z = np.zeros((y_.shape[0], n)); z[:, i] = y_.flatten()
                zeros_mask = np.zeros((y_.shape[0], n)); zeros_mask[:, i] = 1
                xs.append(x_); ys.append(z); masks.append(zeros_mask)
            xs, ys, masks = np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(masks, axis=0)
            xs = self.cuda_tensor(self.x_transformer(xs))
            ys = self.cuda_tensor(self.y_transformer(ys))
            masks = self.cuda_tensor(torch.LongTensor(masks))
            yield xs, (ys, masks)

    def train_test_split_for_multitask(self, test_size):
        assert 0 < test_size < 1, "The test size should be comprize between 0 and 1"
        ts = 0.75
        return self.to_multitask_generator(train_size=ts), self.to_multitask_generator(train_size=-ts)

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