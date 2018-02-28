from .dataset import MetaRegressionDataset, to_categorical
import numpy as np
from os.path import dirname, realpath, join
from os import listdir

DATASETS_ROOT = join(dirname(dirname(realpath(__file__))), 'datasets')


def vectorize_and_pad_sequences(sequences, alpha2int):
    def seq2vec(s):
        i, l, result = 0, 0, [0 for _ in range(len(s))]
        while i < len(s):
            # Is two-letter symbol?
            if alpha2int.get(s[i:i + 2]):
                result[l] = alpha2int[s[i:i + 2]]
                i += 2
            else:
                result[l] = alpha2int[s[i]]
                i += 1
            l += 1
        return result[:l], l
    max_len = max([len(seq) for seq in sequences])

    vectorized_sequences = np.zeros((len(sequences), max_len))
    actual_max_len = 0
    for i, seq in enumerate(sequences):
        res, l = seq2vec(seq)
        vectorized_sequences[i, :l] = res
        actual_max_len = l if l > actual_max_len else actual_max_len
    vectorized_sequences = vectorized_sequences[:, :actual_max_len]
    # print(vectorized_sequences.shape)
    # for row in vectorized_sequences:
    #     print(row.shape)
    final_res = np.array([to_categorical(row, len(alpha2int)+1) for row in vectorized_sequences])
    final_res[:, :, 0] = 0
    return final_res


class MhcIIDatatset(MetaRegressionDataset):
    # refactor the names
    aa_alphabet = list('ARNDCQEGHILKMFPSTWYVBZX*')
    aa_alphabet2int = {el: i + 1 for i, el in enumerate(aa_alphabet)}
    ALPHABET_SIZE = len(aa_alphabet) + 1

    def __init__(self, episode_files, x_transformer=None, y_transformer=None,
                 max_examples_per_episode=None, batch_size=1):
        super(MhcIIDatatset, self).__init__(episode_files, x_transformer, y_transformer, max_examples_per_episode,
                                            batch_size)

        self.x_transformer = lambda x: vectorize_and_pad_sequences(x, self.aa_alphabet2int)
        self.y_transformer = lambda y: np.log(y)

    def episode_loader(self, filename):
        data = np.loadtxt(filename, dtype=str, )
        x, y = data[:, 0], data[:, 1].astype(float).reshape((-1, 1))
        return x, y

    def get_sampling_weights(self):
        episode_sizes = np.log2([len(self.episode_loader(f)[0]) for f in self.episode_files])
        return episode_sizes / np.sum(episode_sizes)


def load_fewshot_mhcII(max_examples_per_episode=20, batch_size=10):
    ds_folder = join(DATASETS_ROOT, 'mhcII_similarity_reduced')
    episode_files = [join(ds_folder, x) for x in listdir(ds_folder)]
    dataset = MhcIIDatatset(episode_files, max_examples_per_episode=max_examples_per_episode, batch_size=batch_size)
    meta_train, meta_test = dataset.train_test_split(test_size=0.25)
    meta_train, meta_valid = meta_train.train_test_split(test_size=0.25)
    meta_test.eval()
    yield meta_train, meta_valid, meta_test


def load_fewshot_mhcII_DRB(max_examples_per_episode=20, batch_size=10):
    ds_folder = join(DATASETS_ROOT, 'mhcII_DRB_all')
    episode_files = [join(ds_folder, x) for x in listdir(ds_folder)]
    sorted(episode_files)

    for test_file in episode_files:
        train_files = episode_files[:]
        train_files.remove(test_file)
        dataset = MhcIIDatatset(train_files, max_examples_per_episode=max_examples_per_episode, batch_size=batch_size)
        meta_test = MhcIIDatatset([test_file], max_examples_per_episode=max_examples_per_episode, batch_size=batch_size)
        meta_train, meta_valid = dataset.train_test_split(test_size=0.25)
        meta_test.eval()
        yield meta_train, meta_valid, meta_test


def load_fewshot_mhcII_DRB_a(max_examples_per_episode=20, batch_size=10):
    ds_folder = join(DATASETS_ROOT, 'mhcII_DRB_all')
    episode_files = [join(ds_folder, x) for x in listdir(ds_folder)]
    sorted(episode_files)
    deb = int(len(episode_files)/2)

    for test_file in episode_files[:deb]:
        train_files = episode_files[:]
        train_files.remove(test_file)
        dataset = MhcIIDatatset(train_files, max_examples_per_episode=max_examples_per_episode, batch_size=batch_size)
        meta_test = MhcIIDatatset([test_file], max_examples_per_episode=max_examples_per_episode, batch_size=batch_size)
        meta_train, meta_valid = dataset.train_test_split(test_size=0.25)
        meta_test.eval()
        yield meta_train, meta_valid, meta_test


def load_fewshot_mhcII_DRB_b(max_examples_per_episode=20, batch_size=10):
    ds_folder = join(DATASETS_ROOT, 'mhcII_DRB_all')
    episode_files = [join(ds_folder, x) for x in listdir(ds_folder)]
    sorted(episode_files)
    deb = int(len(episode_files) / 2)

    for test_file in episode_files[deb:]:
        train_files = episode_files[:]
        train_files.remove(test_file)
        dataset = MhcIIDatatset(train_files, max_examples_per_episode=max_examples_per_episode, batch_size=batch_size)
        meta_test = MhcIIDatatset([test_file], max_examples_per_episode=max_examples_per_episode, batch_size=batch_size)
        meta_train, meta_valid = dataset.train_test_split(test_size=0.25)
        meta_test.eval()
        yield meta_train, meta_valid, meta_test


class BindingdbDatatset(MetaRegressionDataset):
    periodic_elements = ['H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'Al', 'Si', 'P', 'S', 'Cl',
                         'K', 'Ca', 'V', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br',
                         'Nb', 'Mo', 'Tc', 'Ru', 'Pd', 'Ag', 'Sn', 'Sb', 'Te', 'I', 'Gd', 'W',
                         'Re', 'Os', 'Pt', 'Au', 'Hg', 'Bi']

    smilesAlphabet = list('#%)(+*-/.1032547698:=@[]\\cons') + periodic_elements + ['se']
    smilesAlphabet2int = {el: i + 1 for i, el in enumerate(smilesAlphabet)}
    ALPHABET_SIZE = len(smilesAlphabet) + 1
    y_epsilon = 1e-3

    def __init__(self, episode_files, x_transformer=None, y_transformer=None, max_examples_per_episode=None, batch_size=1):
        super(BindingdbDatatset, self).__init__(episode_files, x_transformer, y_transformer, max_examples_per_episode,
                                                batch_size)
        self.x_transformer = lambda x: vectorize_and_pad_sequences(x, self.smilesAlphabet2int)
        self.y_transformer = lambda y: np.log(y+self.y_epsilon)

    def episode_loader(self, filename):
        # print(filename)
        with open(filename, 'r') as f_in:
            lines = [line.split('\t') for line in f_in]
            lines = [(line[0], float(line[1])) for line in lines]
        x, y = zip(*lines)
        x, y = np.array(x), np.array(y).reshape((-1, 1))
        # selec_indixes = [i for i in range(len(x)) if len(x[i]) < 250]
        # return x[selec_indixes], y[selec_indixes]
        return x, y

    def get_sampling_weights(self):
        episode_sizes = np.array([len(self.episode_loader(f)[0]) for f in self.episode_files])
        episode_sizes = np.log2(episode_sizes)
        return episode_sizes/np.sum(episode_sizes)


def load_fewshot_bindingdb(max_examples_per_episode=20, batch_size=10):
    ds_folder = join(DATASETS_ROOT, 'bindingDB')
    episode_files = [join(ds_folder, x) for x in listdir(ds_folder)]
    dataset = BindingdbDatatset(episode_files, max_examples_per_episode=max_examples_per_episode, batch_size=batch_size)
    meta_train, meta_test = dataset.train_test_split(test_size=0.25)
    meta_train, meta_valid = meta_train.train_test_split(test_size=0.25)
    meta_test.eval()
    yield meta_train, meta_valid, meta_test


if __name__ == '__main__':
    # ds_folder = join(DATASETS_ROOT, 'bindingDB')
    # episode_files = [join(ds_folder, x) for x in listdir(ds_folder)]
    # for ep in episode_files:
    #     with open(ep, 'r') as f_in:
    #         lines = [line.split('\t') for line in f_in]
    #         lines = [(line[0], float(line[1])) for line in lines]
    #     x, y = zip(*lines)
    #     x, y = np.array(x), np.array(y).reshape((-1, 1))
    #     print(np.isfinite(np.max(np.log(y+1e-3))) and np.isfinite(np.min(np.log(y+1e-3))))
    # exit()

    from time import time
    t = time()
    ds = load_fewshot_mhcII_DRB()
    for meta_train, meta_valid, meta_test in ds:
        #print(meta_train.__class__, meta_test.__class__, meta_valid.__class__)
        # print(meta_train.ALPHABET_SIZE)
        for episodes in meta_train:
            for ep in episodes:
                print(1)
                # print(ep['name'], ep['Dtest'][0].size(0))
    print("time", time()-t)







