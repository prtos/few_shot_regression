
import os 
import torch
import pickle
import json
import numpy as np
import pandas as pd
from glob import glob
from os.path import dirname, realpath, join, exists
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from metalearn.datasets.metadataset import MetaRegressionDataset
from metalearn.feature_extraction import SequenceTransformer, AdjGraphTransformer
from metalearn.feature_extraction.constants import AMINO_ACID_ALPHABET, SMILES_ALPHABET, ATOM_LIST 
from torch.utils.data import DataLoader, WeightedRandomSampler
from contextlib import contextmanager
from inspect import currentframe, getouterframes

DATASETS_ROOT = join(dirname(dirname(dirname(realpath(__file__)))), 'datasets')


@contextmanager
def let(**bindings):
    frame = getouterframes(currentframe(), 2)[-1][0] # 2 because first frame in `contextmanager` decorator  
    locals_ = frame.f_locals
    original = {var: locals_.get(var) for var in bindings.keys()}
    locals_.update(bindings)
    yield
    locals_.update(original)


class HarmonicsDataset(MetaRegressionDataset):
    def __init__(self, *args, **kwargs):
        super(HarmonicsDataset, self).__init__(*args, **kwargs)
        self.x_transformer = lambda x: torch.FloatTensor(x)
        self.y_transformer = lambda y: torch.FloatTensor(y)
        self.task_descriptor_transformer = lambda z: torch.FloatTensor(z)

    def episode_loader(self, filename):
        metadata = pd.read_csv(filename, nrows=1).values[0]
        data = pd.read_csv(filename, skiprows=2)
        x = data[['x']].values.astype('float32')
        y = data[['y']].values.astype('float32')
        return x, y, None


class MhcDataset(MetaRegressionDataset):
    ALPHABET_SIZE = len(AMINO_ACID_ALPHABET) + 1

    def __init__(self, *args, **kwargs):
        super(MhcDataset, self).__init__(*args, **kwargs)

        x_transformer = SequenceTransformer(AMINO_ACID_ALPHABET)
        self.x_transformer = lambda x: x_transformer.transform(x)
        self.y_transformer = lambda y: torch.FloatTensor(y)
        self.task_descriptor_transformer = None
        self.max_test_examples = 10000
        self.vocab = AMINO_ACID_ALPHABET

    def episode_loader(self, filename):
        data = np.loadtxt(filename, dtype=str, )
        x, y = data[:, 0], data[:, 1].astype(float).reshape((-1, 1))
        scaler = MinMaxScaler()
        y = scaler.fit_transform(y).astype('float32')
        return x, y, None

    def get_sampling_weights(self):
        episode_sizes = np.log2([len(self.episode_loader(f)[0]) for f in self.tasks_filenames])
        return episode_sizes / np.sum(episode_sizes)


class ChemblDataset(MetaRegressionDataset):
    def __init__(self, *args, use_graph=True, **kwargs):
        super(ChemblDataset, self).__init__(*args, **kwargs)
        y_epsilon = 1e-7
        if use_graph:
            transformer = AdjGraphTransformer()
        else:
            transformer = SequenceTransformer(SMILES_ALPHABET, returnTensor=True)
        prot_transformer = SequenceTransformer(AMINO_ACID_ALPHABET)
        self.x_transformer = lambda x: transformer.transform(x)
        self.y_transformer = lambda y: torch.FloatTensor(y)
        self.task_descr_transformer = lambda z: None if z is None else prot_transformer.transform([z])[0] 
        self.max_test_examples = 500
        self.vocab = SMILES_ALPHABET if not use_graph else ATOM_LIST 
        
    def episode_loader(self, filename):
        with open(filename, 'r') as f_in:
            f_in.readline() # reand and neglect the first line
            protein = f_in.readline()[:-1].upper()
        data = pd.read_csv(filename, header=None, skiprows=2, delim_whitespace=True)
        data = data.values
        x, y = data[:, 1], data[:, 2].astype('float').reshape((-1, 1))
        scaler = MinMaxScaler()
        y = scaler.fit_transform(y).astype('float32')
        return x, y, None

    def get_sampling_weights(self):
        episode_sizes = np.array([len(self.episode_loader(f)[0]) for f in self.tasks_filenames])
        episode_sizes = np.log2(episode_sizes)
        return episode_sizes/np.sum(episode_sizes)


class PubchemToxDataset(ChemblDataset):
    def __init__(self, *args, **kwargs):
        super(PubchemToxDataset, self).__init__(*args, **kwargs)
        def file_len(fname):
            with open(fname) as f:
                for i, _ in enumerate(f):
                    pass
            return i+1
        self.tasks_sizes = np.array([file_len(f) for f in self.tasks_filenames])

    def episode_loader(self, filename):
        df = pd.read_csv(filename, skiprows=1)
        data = df[df.iloc[:, 0].str.len() <= 300].values
        if data.shape[0] == 0:
            data = df.sample(min(300, df.shape[0]), replace=False).values
        x, y = data[:, 0], data[:, 1].astype('float').reshape((-1, 1))
        scaler = MinMaxScaler()
        y = scaler.fit_transform(y).astype('float32')
        return x, y, None

    def get_sampling_weights(self):
        temp = np.log(self.tasks_sizes)
        temp /= np.sum(temp)
        return temp


class Tox21Dataset(ChemblDataset):
    def episode_loader(self, filename):
        data = pd.read_csv(filename, header=None,  delim_whitespace=True).values
        x, y = data[:, 0], data[:, 1].astype('float').reshape((-1, 1))
        return x, y, None

def __get_partitions(dataset_cls, episode_files, batch_size, test_files=None, 
                     test_size=0.25, valid_size=0.25, **kwargs):
    if test_files is not None:
        train_files = episode_files
    else:
        train_files, test_files = train_test_split(episode_files, test_size=test_size)
    train_files, valid_files = train_test_split(train_files, test_size=valid_size)
    train = dataset_cls(train_files, **kwargs)
    valid = dataset_cls(valid_files, **kwargs)
    test = dataset_cls(test_files, is_test=True, **kwargs)
    collate = lambda x: list(zip(*x))
    train = DataLoader(train, batch_size=batch_size, collate_fn=collate, 
                    sampler=WeightedRandomSampler(np.log(train.task_sizes()), len(train)))
    test = DataLoader(test, batch_size=batch_size, collate_fn=collate)
    valid = DataLoader(valid, batch_size=batch_size, collate_fn=collate)
    return train, valid, test


def load_dataset(dataset_name, ds_folder=None, max_tasks=None, fold=None, **kwargs):
    if ds_folder is None:
        ds_folder = join(DATASETS_ROOT, dataset_name)
    
    maps = dict(
        pubchemtox=dict(files=[join(ds_folder, x) for x in glob(f"{ds_folder}/*/*.csv")], 
                     dscls=PubchemToxDataset),
        mhc=dict(files=[join(ds_folder, x) for x in os.listdir(ds_folder)], 
                     dscls=MhcDataset),
        easytoy=dict(files=[join(ds_folder, x) for x in os.listdir(ds_folder) if x.endswith('.csv')], 
                     dscls=HarmonicsDataset),
        tox21=dict(files=[join(ds_folder, x) for x in os.listdir(ds_folder) if x.endswith('.smiles')], 
                     dscls=Tox21Dataset),
        chembl=dict(files=[join(ds_folder, x) for x in os.listdir(ds_folder) if x.endswith('.tsv')], 
                     dscls=ChemblDataset),
    )

    if dataset_name not in maps:
        raise Exception(f"Unhandled dataset. The name of \
            the dataset should be one of those: {list(maps.keys())}")
    dataset_cls, episode_files = maps[dataset_name]['dscls'], maps[dataset_name]['files'][:max_tasks]
    test_files = None
    if fold is not None:
        episode_files = sorted(episode_files)
        test_files = [episode_files[fold]]
        del episode_files[fold]

    splitting_file = join(ds_folder, dataset_name+'.json')
    if exists(splitting_file):
        with open(splitting_file) as fd:
            temp = json.load(fd)
        episode_files = [join(dirname(ds_folder), f) for f in temp['Dtrain']]
        test_files = [join(dirname(ds_folder), f) for f in temp['Dtest']]

    return __get_partitions(dataset_cls, episode_files, test_files=test_files, **kwargs)


if __name__ == '__main__':
    from time import time
    t = time()
    ds = load_dataset('pubchemtox', batch_size=32, max_examples_per_episode=10 )
    train, valid, test = ds
    all_files = train.dataset.tasks_filenames + valid.dataset.tasks_filenames + test.dataset.tasks_filenames
    # for f in all_files:
    #     x, y, _ = sum([len(train.dataset.episode_loader(f)[0]) > 100 for f in all_files], 0)
    #     print(len(x))
    lens_0 = [len(train.dataset.episode_loader(f)[0]) for f in all_files]
    lens = lens_0[:]
    print(f'sans filtre {len(lens)} {sum(lens)}')
    lens = [l>50 for l in lens_0]
    print(f'avec filtre 50 {len(lens)} {sum(lens)}')
    lens = [l>100 for l in lens_0]
    print(f'avec filtre 100 {len(lens)} {sum(lens)}')
    print("time", time()-t)
