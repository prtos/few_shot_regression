import os
import numpy as np
import torch
from keras.utils.np_utils import to_categorical
from PIL import Image


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
        fx = self.x_transformer if self.x_transformer is not None else (lambda x: x)
        fy = self.y_transformer if self.y_transformer is not None else (lambda x: x)
        xtrain, ytrain = fx(xtrain), fy(ytrain)
        xtest, ytest = fx(xtest), fy(ytest)
        x_test, y_test = MetaDataset.to_tensors(xtest, ytest, self.use_available_gpu)
        return dict(Dtrain=MetaDataset.to_tensors(xtrain, ytrain, self.use_available_gpu),
                    Dtest=x_test,
                    name=MetaDataset.str_to_tensor(name, self.use_available_gpu)), y_test

    @staticmethod
    def to_tensors(x, y, use_available_gpu=True):
        tensor_x, tensor_y = torch.LongTensor(x), torch.FloatTensor(y)
        if torch.cuda.is_available() and use_available_gpu:
            tensor_x, tensor_y = tensor_x.cuda(), tensor_y.cuda()
        # print(tensor_y.size())
        return tensor_x, tensor_y

    @staticmethod
    def str_to_tensor(s, use_available_gpu=True):
        res = torch.IntTensor([ord(char) for char in s])
        if torch.cuda.is_available() and use_available_gpu:
            res = res.cuda()
        return res


class MetaRegressionDataset(MetaDataset):
    TRAIN, TEST = 0, 1

    def __init__(self, episode_files, x_transformer=None, y_transformer=None, max_examples_per_episode=None,
                 batch_size=1):
        super(MetaRegressionDataset, self).__init__(x_transformer, y_transformer, batch_size)
        self.episode_files = episode_files[:]
        self.max_examples_per_episode = max_examples_per_episode
        self.mode = self.TRAIN

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
        if self.max_examples_per_episode is None:
            half_size = int(size_episode / 2)
            train_idx, test_idx = idx[:half_size], idx[half_size:]
        elif size_episode <= self.max_examples_per_episode*2:
            half_size = int(size_episode / 2)
            train_idx, test_idx = idx[:half_size], idx[half_size:]
        else:
            n = self.max_examples_per_episode
            train_idx, test_idx = idx[:n], idx[n:2*n]
            if self.mode == self.TEST:
                train_idx, test_idx = idx[:n], idx[n:]
        # print('idx', train_idx, test_idx)
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
                # episode_filenames = choices(self.episode_files, sampling_weights, k=self.batch_size)
                x_batch, y_batch = zip(*[self.__filename2episode(epf) for epf in episode_filenames])
                yield x_batch, torch.stack(y_batch)

    def full_datapoints_generator(self):
        batch_size = self.batch_size * self.max_examples_per_episode
        xs, ys = zip(*[self.episode_loader(epfile) for epfile in self.episode_files])
        x_ = np.concatenate(xs, axis=0)
        y_ = np.concatenate(ys, axis=0)
        for x, y in make_generator(x_, y_, batch_size):
            yield MetaDataset.to_tensors(self.x_transformer(x), self.y_transformer(y))

    def full_datapoints(self):
        xs, ys = zip(*[self.episode_loader(epfile) for epfile in self.episode_files])
        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        return MetaDataset.to_tensors(self.x_transformer(x), self.y_transformer(y))


class MetaClassificationDataset(MetaDataset):
    def __init__(self, classe_dirs, x_transformer=None, y_transformer=None, nb_classes_per_episode=5,
                 nb_examples_per_class=None, max_episodes_produced=None, infinite_generator=True):
        super(MetaClassificationDataset, self).__init__(x_transformer, y_transformer,
                                                        max_episodes_produced, infinite_generator)
        self.classe_dirs = classe_dirs[:]
        self.nb_classes_per_episode = nb_classes_per_episode
        self.nb_examples_per_class = nb_examples_per_class

    def train_test_split(self, test_size):
        if test_size < 1:
            test_size = int(len(self.classe_dirs) * test_size)
        else:
            test_size = int(test_size)
        assert test_size < len(self.classe_dirs), "The test size should be lower than the number of episodes"

        np.random.shuffle(self.classe_dirs)
        metatest_class_dirs = self.classe_dirs[:test_size]
        metatrain_class_dirs= self.classe_dirs[test_size:]


        metatrain = MetaClassificationDataset(metatrain_class_dirs, self.x_transformer, self.y_transformer,
                                              self.nb_classes_per_episode, self.nb_examples_per_class,
                                              self.max_episodes_produced, self.infinite_generator)
        metatest = MetaClassificationDataset(metatest_class_dirs, self.x_transformer, self.y_transformer,
                                              self.nb_classes_per_episode, self.nb_examples_per_class,
                                              self.max_episodes_produced, self.infinite_generator)
        return metatrain, metatest

    def __iter__(self):
        N = len(self.classe_dirs)
        max_episodes_produced = max(N, self.max_episodes_produced)
        np.random.shuffle(self.classe_dirs)
        keep_looping = True
        while keep_looping:
            for i in range(0, max_episodes_produced, self.batch_size):
                batch = []
                for j in range(i, i+self.batch_size):
                    np.random.shuffle(self.classe_dirs)
                    classes_dirs_for_episode = self.classe_dirs[:self.nb_classes_per_episode]
                    temp = [os.listdir(dir)[:2*self.nb_examples_per_class] for dir in classes_dirs_for_episode]
                    for l in temp:
                        np.random.shuffle(l)
                    train_files = [l[:self.nb_examples_per_class] for l in temp]
                    test_files = [l[self.nb_examples_per_class:] for l in temp]
                    x_train, y_train = load_images_data(train_files)
                    x_test, y_test = load_images_data(test_files)
                    batch.append(self._episode(x_train, y_train, x_test, y_test))
                yield batch
            keep_looping = self.infinite_generator


if __name__ == '__main__':
    pass