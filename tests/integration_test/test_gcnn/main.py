import torch
import numpy as np
import pandas as pd
from few_shot_regression.utils.feature_extraction.graph_cnn import GraphCnnFeaturesExtractor
from few_shot_regression.utils.feature_extraction.cnn import Cnn1dFeaturesExtractor
from few_shot_regression.utils.feature_extraction.transformers import *
from few_shot_regression.models.maml import Regressor
from pytoune.framework import Model
from torch.optim import Adam
from torch.nn import MSELoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def make_generator(x, y, batch_size=-1, infinite=True):
    if batch_size == -1:
        while True:
            yield x, y
    else:
        loop = 0
        while loop < 1 or infinite:
            for i in range(0, len(y), batch_size):
                yield x[i:i+batch_size], y[i:i+batch_size]
            loop += 1


def get_fragments_vocab(vocab_file, molecules_list=None):
    """
    This function returns the vocab of the fragments as a list of smiles given a list of molecule.
    It is expensive to generate the vocab for the first time so we highly recommend you to
    provide the vocab_file so we can save the results for you and can use it later.
    :param vocab_file: A filename that already contains the vocab if it has been pre-generated,
    or a filename that will be used to store the new voab generated.
    :param molecules_list: list of molecules from which the vocan is extracted
    :return: list of fragment's smiles
    """
    if os.path.exists(vocab_file):
        with open(vocab_file, 'r') as fd:
            vocab = [el for el in fd]
    else:
        assert molecules_list is not None, "the molecule list should not be empty if the vocab_file doesn't exist"
        vocab = generate_fragments_vocab(molecules_list, vocab_file)
    return vocab


if __name__ == '__main__':
    import os
    vocab_file = 'vocab.txt'
    data_file = 'cep-processed.csv'
    if not os.path.exists(vocab_file):
        with open(data_file) as fd:
            _ = fd.readline()
            molecules_list = [Chem.MolFromSmiles(s.split(',')[0]) for s in fd]
            vocab = get_fragments_vocab(vocab_file, molecules_list)
    else:
        with open(data_file) as fd:
            vocab = [l[:-1] for l in fd]

    batch_size = 32
    data = pd.read_csv(data_file).values
    x = data[:, 0]
    y = data[:, 1].astype("float")[:, np.newaxis]
    y = (y - y.mean())/y.std()
    print(y.max(), y.min())
    print(np.mean((y - y.mean())**2), np.mean(np.abs(y - y.mean())))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25)
    step_train = int(len(y_train)/batch_size)
    step_valid = int(len(y_valid) / batch_size)
    # transformer = MolecularGraphTransformer()
    # feature_extractor = GraphCnnFeaturesExtractor(1 + len(MOL_ALPHABET), 20, [50, 50], normalize_features=False)
    transformer = MolecularTreeDecompositionTransformer(vocab)
    feature_extractor = GraphCnnFeaturesExtractor(1 + len(MOL_ALPHABET), 20, [50, 50], normalize_features=False)
    # transformer = SequenceTransformer(SMILES_ALPHABET)
    # feature_extractor = Cnn1dFeaturesExtractor(1 + len(SMILES_ALPHABET), 20, [50, 50], 3,
    #                                            pooling_len=2, dilatation_rate=2, normalize_features=False)

    x_train, y_train = transformer.transform(x_train), torch.FloatTensor(y_train)
    x_valid, y_valid = transformer.transform(x_valid), torch.FloatTensor(y_valid)
    x_test, y_test = transformer.transform(x_test), torch.FloatTensor(y_test)
    g_train = make_generator(x_train, y_train, batch_size=batch_size)
    g_valid = make_generator(x_valid, y_valid, batch_size=128)
    g_test = make_generator(x_test, y_test, batch_size=128, infinite=False)
    print(y_train.max(), y_train.min(), type(y_train))

    network = Regressor(feature_extractor, 1)
    model = Model(network, Adam(network.parameters(), lr=1e-2), MSELoss())
    model.fit_generator(g_train, g_valid, steps_per_epoch=step_train, validation_steps=step_valid)
    y_pred = model.predict_generator(g_test)
    print('r2 test', r2_score(y_test, y_pred))
