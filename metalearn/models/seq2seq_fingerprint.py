import torch
import numpy as np
import torch.nn.functional as F
import deepchem as dc
import deepchem.models.tensorgraph.optimizers as dcopt
import logging
from torch.nn.utils.rnn import pack_padded_sequence
from torch import nn
from torch.optim import Adam
from pytoune.framework import Model
from sklearn.model_selection import GridSearchCV, train_test_split
from .base import MetaLearnerRegression, FeaturesExtractorFactory, MetaNetwork
from .krr import KrrLearner, KrrLearnerCV
from .utils import reset_BN_stats, to_unit
from .fp_learner import algos_classes, algos_grid

logger = logging.getLogger('deepchem.models.tensorgraph.tensor_graph')
logger.setLevel(logging.DEBUG)
# Add a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def generate_sequences(data, epochs):
    n = len(data)
    if n < 10000:
        for i in range(epochs):
            for s in data:
                yield (s, s)
    else:
        k = 10000
        for j in range(0, n, k):
            for s in data[j:j+k]:
                yield (s, s)


MAX_LENGTH = 300

def train_seqtoseq(train_data, embedding_dimension, tokens, max_length, encoder_layers=1, 
                 decoder_layers=1, dropout=0.1, tb_folder='fingerprint', 
                 batch_size=32, n_epochs=100, steps_per_epoch=None):
        train_generator = generate_sequences(train_data, n_epochs)
        model = dc.models.SeqToSeq(tokens, tokens, max_length,
                                    encoder_layers=encoder_layers,
                                    decoder_layers=decoder_layers,
                                    embedding_dimension=embedding_dimension,
                                    batch_size=batch_size,
                                    verbose=True,
                                    tensorboard=True, 
                                    tensorboard_log_frequency=1,
                                    model_dir=tb_folder)
        if steps_per_epoch is None:
            steps_per_epoch = len(train_data)/model.batch_size 

        model.set_optimizer(
            dcopt.Adam(learning_rate=dcopt.ExponentialDecay(0.004, 0.95, steps_per_epoch)))
        model.fit_sequences(train_generator, checkpoint_interval=steps_per_epoch)
        return model

def filter_long(x, y):
    res = list(zip(*[(a, b) for a, b in zip(x, y) if len(a) <= MAX_LENGTH]))
    return np.array(res[0]), np.array((res[1]))

def fit_and_eval(model_seq2seq, episode, algo):
    x_train, y_train = filter_long(*episode['Dtrain'])
    x_train = model_seq2seq.predict_embeddings(x_train)
    x_test, y_test = filter_long(*episode['Dtest'])
    x_test = model_seq2seq.predict_embeddings(x_test)
    train_size = len(x_train)
    model_cls = algos_classes[algo]
    param_grid = algos_grid[algo]
    model = model_cls(**param_grid)
    if algo in ["gb", "rf"]:
        model = model_cls(**param_grid)
    else:
        model = GridSearchCV(model_cls(), param_grid, cv=train_size, refit=True, n_jobs=-1)
    model.fit(x_train, y_train.ravel())
    return y_test, model.predict(x_test)
        

class Seq2SeqLearner:
    def __init__(self, embedding_dim, encoder_layers=1, decoder_layers=1, dropout=0.0, algo='gb'):
        self.embedding_dim=embedding_dim
        self.encoder_layers=encoder_layers
        self.decoder_layers=decoder_layers
        self.dropout=dropout
        self.algo = algo

    def fit(self, meta_train, meta_valid, n_epochs=100, steps_per_epoch=100,
            log_filename=None, checkpoint_filename=None, tboard_folder=None):
        seqs = list(set(
                    sum(([meta_train.dataset.episode_loader(f)[0].tolist()
                        for f in meta_train.dataset.tasks_filenames] + 
                        [meta_valid.dataset.episode_loader(f)[0].tolist()
                        for f in meta_valid.dataset.tasks_filenames]),
                        [])))                              
        seqs = [el for el in seqs if len(el) <= MAX_LENGTH]
        train_seqs, valid_seqs = train_test_split(seqs, test_size=0.1)
        
        tokens = list(set("".join(train_seqs)))
        max_length = min(max([len(el) for el in seqs]), MAX_LENGTH)
        print(f"Train size {len(train_seqs)}")
        print(f"Valid size {len(valid_seqs)}")

        self.model = train_seqtoseq(train_seqs, self.embedding_dim, tokens, max_length, 
            encoder_layers=self.encoder_layers, decoder_layers=self.decoder_layers, 
            dropout=self.dropout, tb_folder=tboard_folder, 
            batch_size=meta_train.batch_size, n_epochs=n_epochs)

        predicted = self.model.predict_from_sequences(valid_seqs, beam_width=1)
        acc = sum([''.join(p) == s for s,p in zip(valid_seqs, predicted)])
        acc = 1.0*acc/len(valid_seqs)
        print('Valid performance', acc)
            
    def evaluate(self, metatest, metrics=[F.mse_loss], **kwargs):
        metatest.dataset.raw_inputs = True
        assert len(metrics) >= 1, "There should be at least one valid metric in the list of metrics "
        metrics_per_dataset = {metric.__name__: {} for metric in metrics}
        metrics_per_dataset["size"] = dict()
        metrics_per_dataset["name"] = dict()
        for episodes in metatest:
            for (episode, _) in zip(*episodes):
                y_test, y_pred = fit_and_eval(self.model, episode, self.algo)
                y_pred = torch.Tensor(y_pred.flatten())
                y_test = torch.Tensor(y_test.flatten())
                ep_idx = episode['idx']
                ep_name_is_new = (ep_idx not in metrics_per_dataset["size"])
                for metric in metrics:
                    m_value = to_unit(metric(y_pred, y_test))
                    if ep_name_is_new:
                        metrics_per_dataset[metric.__name__][ep_idx] = [m_value]
                    else:
                        metrics_per_dataset[metric.__name__][ep_idx].append(m_value)
                metrics_per_dataset['size'][ep_idx] = y_test.size(0)
                metrics_per_dataset['name'][ep_idx] = metatest.dataset.tasks_filenames[ep_idx]

        return metrics_per_dataset
