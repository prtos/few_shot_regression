import pickle
import pandas as pd
from os.path import dirname, realpath, join
from os import listdir
from metalearn.datasets.metadataset import MetaRegressionDataset
from metalearn.feature_extraction.transformers import *

DATASETS_ROOT = join(dirname(dirname(dirname(realpath(__file__)))), 'datasets')


class HarmonicsDatatset(MetaRegressionDataset):
    def __init__(self, *args, **kwargs):
        super(HarmonicsDatatset, self).__init__(*args, **kwargs)
        self.x_transformer = lambda x: torch.FloatTensor(x)
        self.y_transformer = lambda y: torch.FloatTensor(y)
        self.task_descriptor_transformer = lambda z: torch.FloatTensor(z)

    def episode_loader(self, filename):
        metadata = pd.read_csv(filename, nrows=1).values[0]
        data = pd.read_csv(filename, skiprows=2)
        x = data[['x']].values.astype('float32')
        y = data[['y']].values.astype('float32')
        return x, y, None


class MovielensDatatset(MetaRegressionDataset):

    def __init__(self, *args, **kwargs):
        super(MovielensDatatset, self).__init__(*args, **kwargs)
        # features of the movies
        with open(join(DATASETS_ROOT, 'movielens/movies_features.pkl'), 'rb') as f:
            self.features_movies = pickle.load(f)

        self.x_transformer = lambda x: torch.FloatTensor(x)
        self.y_transformer = lambda y: torch.FloatTensor(y)
        self.task_descriptor_transformer = lambda z: torch.FloatTensor(z)

    def episode_loader(self, filename):
        data = pd.read_csv(filename, sep='\t')
        temp = data.as_matrix()
        x, y = temp[:, 0], temp[:, 1].reshape((-1, 1))
        x = np.array([self.features_movies[movie] for movie in x])
        return x, y, None


class UciSelectionDatatset(MetaRegressionDataset):
    def __init__(self, *args, **kwargs):
        super(UciSelectionDatatset, self).__init__(*args, **kwargs)
        self.x_transformer = lambda x: torch.FloatTensor(x)
        self.y_transformer = lambda y: torch.FloatTensor(y)
        self.task_descriptor_transformer = lambda z: torch.FloatTensor(z)

    def episode_loader(self, filename):
        metadata = pd.read_csv(filename, nrows=1).values[0]
        data = pd.read_csv(filename, skiprows=2)
        x = data[['param_C', 'param_gamma']].values
        x = np.log(x)
        y = data[['mean_test_acc', 'std_test_acc']].values[:, 0:1]
        return x, y, metadata


class MhcDatatset(MetaRegressionDataset):
    ALPHABET_SIZE = len(AMINO_ACID_ALPHABET) + 1

    def __init__(self, *args, **kwargs):
        super(MhcDatatset, self).__init__(*args, **kwargs)

        x_transformer = SequenceTransformer(AMINO_ACID_ALPHABET)
        self.x_transformer = lambda x: x_transformer.transform(x)
        self.y_transformer = lambda y: torch.FloatTensor(np.log(y))
        self.task_descriptor_transformer = None

    def episode_loader(self, filename):
        data = np.loadtxt(filename, dtype=str, )
        x, y = data[:, 0], data[:, 1].astype(float).reshape((-1, 1))
        return x, y

    def get_sampling_weights(self):
        episode_sizes = np.log2([len(self.episode_loader(f)[0]) for f in self.tasks_filenames])
        return episode_sizes / np.sum(episode_sizes)


class BindingdbDatatset(MetaRegressionDataset):
    def __init__(self, *args, **kwargs):
        super(BindingdbDatatset, self).__init__(*args, **kwargs)

    def episode_loader(self, filename):
        with open(filename, 'r') as f_in:
            protein = f_in.readline()[:-1].upper()
            # measurement = f_in.readline()
        data = pd.read_csv(filename, header=None, skiprows=2, delim_whitespace=True).values
        x, y = data[:, 0], data[:, 1].astype('float').reshape((-1, 1))
        return x, y, protein

    def get_sampling_weights(self):
        episode_sizes = np.array([len(self.episode_loader(f)[0]) for f in self.tasks_filenames])
        episode_sizes = np.log2(episode_sizes)
        return episode_sizes/np.sum(episode_sizes)


def load_episodic_bindingdb(use_graph_for_mol=False, max_examples_per_episode=20, batch_size=10, ds_folder=None):
    y_epsilon = 1e-3
    if ds_folder is None:
        ds_folder = join(DATASETS_ROOT, 'bindingDB')
    episode_files = [join(ds_folder, x) for x in listdir(ds_folder)][:1000]
    if use_graph_for_mol:
        transformer = MolecularGraphTransformer(returnTensor=True, return_adj_matrix=False)
    else:
        transformer = SequenceTransformer(SMILES_ALPHABET, returnTensor=True)
    prot_transformer = SequenceTransformer(AMINO_ACID_ALPHABET)
    x_transformer = lambda x: transformer.transform(x)
    y_transformer = lambda y: torch.FloatTensor(np.log(y + y_epsilon))
    task_descr_transformer = lambda z: prot_transformer.transform([z])[0]
    dataset = BindingdbDatatset(episode_files, x_transformer, y_transformer, task_descr_transformer,
                                max_examples_per_episode=max_examples_per_episode,
                                batch_size=batch_size)
    meta_train, meta_test = dataset.train_test_split(test_size=0.25)
    meta_test.eval()
    return meta_train, meta_test


def load_episodic_movielens(max_examples_per_episode=20, batch_size=10, ds_folder=None):
    if ds_folder is None:
        ds_folder = join(DATASETS_ROOT, 'movielens')
    episode_files = [join(ds_folder, x) for x in listdir(ds_folder) if x.endswith('.txt')]
    dataset = MovielensDatatset(episode_files, max_examples_per_episode=max_examples_per_episode, batch_size=batch_size)
    meta_train, meta_test = dataset.train_test_split(test_size=1/6.0)
    meta_test.eval()
    return meta_train, meta_test


def load_episodic_uciselection(max_examples_per_episode=20, batch_size=10):
    ds_folder = join(DATASETS_ROOT, 'uci_rbf')
    episode_files = [join(ds_folder, x) for x in listdir(ds_folder) if x.endswith('.csv')][:400]
    dataset = UciSelectionDatatset(episode_files, max_examples_per_episode=max_examples_per_episode, batch_size=batch_size)
    meta_train, meta_test = dataset.train_test_split(test_size=1/4.0)
    meta_test.eval()
    print(len(meta_train), len(meta_test))
    return meta_train, meta_test


def load_episodic_harmonics(max_examples_per_episode=20, batch_size=10, ds_folder=None):
    if ds_folder is None:
        ds_folder = join(DATASETS_ROOT, 'toy')
    episode_files = [join(ds_folder, x) for x in listdir(ds_folder) if x.endswith('.csv')]
    dataset = HarmonicsDatatset(episode_files, max_examples_per_episode=max_examples_per_episode,
                                batch_size=batch_size, max_test_examples=1000)
    meta_train, meta_test = dataset.train_test_split(test_size=1/5.0)
    meta_test.eval()
    print(len(meta_train), len(meta_test))
    return meta_train, meta_test


def load_episodic_easyharmonics(max_examples_per_episode=20, batch_size=10, ds_folder=None):
    if ds_folder is None:
        ds_folder = join(DATASETS_ROOT, 'toy_easy')
    episode_files = [join(ds_folder, x) for x in listdir(ds_folder) if x.endswith('.csv')][:1000]
    dataset = HarmonicsDatatset(episode_files, max_examples_per_episode=max_examples_per_episode,
                                batch_size=batch_size, max_test_examples=1000)
    meta_train, meta_test = dataset.train_test_split(test_size=1/5.0)
    meta_test.eval()
    print(len(meta_train), len(meta_test))
    return meta_train, meta_test


def load_episodic_mhc(max_examples_per_episode=20, batch_size=10, ds_folder=None):
    if ds_folder is None:
        ds_folder = join(DATASETS_ROOT, 'mhc_all')
    episode_files = [join(ds_folder, x) for x in listdir(ds_folder)]

    dataset = MhcDatatset(episode_files, max_examples_per_episode=max_examples_per_episode, batch_size=batch_size)
    meta_train, meta_test = dataset.train_test_split(test_size=1/4.0)
    meta_test.eval()
    return meta_train, meta_test


if __name__ == '__main__':
    from time import time
    t = time()
    ds = load_episodic_mhc()
    for meta_train, meta_valid, meta_test in ds:
        for episodes in meta_train:
            for ep in episodes:
                print(ep[0])
                exit()
    print("time", time()-t)







