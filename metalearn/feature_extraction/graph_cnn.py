from torch.nn import ReLU, Embedding, ModuleList, BatchNorm1d
from metalearn.feature_extraction.common_modules import *
import torch.nn.functional as F
from rdkit import Chem
from itertools import zip_longest
import numpy as np


class GraphConvLayer(Module):
    """This is the representation of algorithm 2 in
    http://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints.pdf
    """

    def __init__(self, input_size, kernel_size, padding_index=0):
        super(GraphConvLayer, self).__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.b_norm = BatchNorm1d(kernel_size)
        self.linear = Linear(input_size, kernel_size)

    def forward(self, x):
        """
        :param x: Graph representation of the molecules (must be nodes' type, edges matrix)
        :return:
        """
        assert type(x) == tuple, 'must be a tuple of matrix'
        nodes_features, nodes_neighbors_indexes = x
        assert nodes_neighbors_indexes.size(0) == nodes_features.size(0)

        n, m = nodes_features.size()
        # make index 0 the padding index
        temp = torch.cat([torch.zeros((1, m)), nodes_features], dim=0)
        # and shift every node index accordingly
        indexes = nodes_neighbors_indexes + 1
        temp = torch.index_select(temp, dim=0, index=indexes.view(-1))
        neighbors_features = temp.view((*nodes_neighbors_indexes.size(), -1))

        temp = nodes_features + neighbors_features.sum(dim=1)  # ligne 8
        nodes_new_features = F.relu(self.b_norm(self.linear(temp)))  # ligne 9
        return nodes_new_features, nodes_neighbors_indexes


class GraphCnnFeaturesExtractor(ClonableModule):
    def __init__(self, vocab_size, embedding_size, kernel_sizes, normalize_features=True):
        super(GraphCnnFeaturesExtractor, self).__init__()

        super(GraphCnnFeaturesExtractor, self).__init__()
        self.vocab_size = vocab_size
        self.kernel_sizes = kernel_sizes
        self.node_type_embedding_size = embedding_size
        self.node_type_embedding_layer = Embedding(self.vocab_size, self.node_type_embedding_size, padding_idx=0)
        self.normalize_output = normalize_features

        input_size = self.node_type_embedding_size
        self.gc_layers = ModuleList()
        for kernel_size in kernel_sizes:
            self.gc_layers.append(GraphConvLayer(input_size, kernel_size))
            input_size = kernel_size

        if self.normalize_output:
            self.norm_layer = UnitNormLayer()

    def forward(self, batch_x):
        """
        :param x: Graph representation of the molecules, must be a tuple of (nodes_type, neighbors_indexes)
        neighbors_indexes should have -1 as padding indexes
        :return:
        """
        assert isinstance(batch_x, (list, tuple)), 'must be a tuple or a list of MolecularGraph'
        batch_size = len(batch_x)
        nodes_type, neighbors_indexes, nb_nodes_per_mol, n = [], [], [], 0
        for i in range(batch_size):
            molsize = batch_x[i][0].size(0)
            nodes_type.append(batch_x[i][0])
            neighbors_indexes.append(n + batch_x[i][1])
            nb_nodes_per_mol.append(molsize)
            n += molsize
        nodes_type = torch.cat(nodes_type, dim=0)
        neighbors_indexes = torch.cat(neighbors_indexes, dim=0)
        type_embedding = self.node_type_embedding_layer(nodes_type)

        nodes_features = type_embedding
        for i in range(len(self.kernel_sizes)):
            nodes_features, neighbors_indexes = self.gc_layers[i]((nodes_features, neighbors_indexes))

        phis = torch.stack([torch.max(mol_temp, dim=0)[0]
                            for mol_temp in torch.split(nodes_features, nb_nodes_per_mol, dim=0)], dim=0)

        if self.normalize_output:
            phis = self.norm_layer(phis)

        return phis

    @property
    def output_dim(self):
        return self.kernel_sizes[-1]


class GraphCnnFeaturesDavidDuvenauExtractor(ClonableModule):
    def __init__(self, vocab_size, embedding_size, kernel_sizes, output_size, normalize_features=True):
        super(GraphCnnFeaturesDavidDuvenauExtractor, self).__init__()
        self.vocab_size = vocab_size
        self.kernel_sizes = kernel_sizes
        self.output_size = output_size
        self.node_type_embedding_size = embedding_size
        self.node_type_embedding_layer = Embedding(self.vocab_size, self.node_type_embedding_size, padding_idx=0)
        self.normalize_output = normalize_features

        input_size = self.node_type_embedding_size
        self.gc_layers = ModuleList()
        self.smoothness_layers = ModuleList()
        for kernel_size in kernel_sizes:
            self.gc_layers.append(GraphConvLayer(input_size, kernel_size))
            self.smoothness_layers.append(Linear(kernel_size, output_size))
            input_size = kernel_size

        if self.normalize_output:
            self.norm_layer = UnitNormLayer()

    def forward(self, batch_x):
        """
        :param x: Graph representation of the molecules, must be a tuple of (nodes_type, neighbors_indexes)
        neighbors_indexes should have -1 as padding indexes
        :return:
        """
        assert isinstance(batch_x, (list, tuple)), 'must be a tuple or a list of MolecularGraph'
        batch_size = len(batch_x)
        nodes_type, neighbors_indexes, nb_nodes_per_mol, n = [], [], [], 0
        for i in range(batch_size):
            molsize = batch_x[i][0].size(0)
            nodes_type.append(batch_x[i][0])
            neighbors_indexes.append(n + batch_x[i][1])
            nb_nodes_per_mol.append(molsize)
            n += molsize
        nodes_type = torch.cat(nodes_type, dim=0)
        neighbors_indexes = torch.cat(neighbors_indexes, dim=0)
        type_embedding = self.node_type_embedding_layer(nodes_type)

        nodes_features = type_embedding
        fingerprints = []
        for i in range(len(self.kernel_sizes)):
            nodes_features, neighbors_indexes = self.gc_layers[i]((nodes_features, neighbors_indexes))
            temp = self.smoothness_layers[i](nodes_features)
            temp = F.softmax(temp, dim=1)

            temp = torch.stack([torch.max(mol_temp, dim=0)[0] for mol_temp in torch.split(temp, nb_nodes_per_mol, dim=0)], dim=0)
            fingerprints.append(temp)

        fingerprints = torch.stack(fingerprints, dim=0)
        phis = torch.max(fingerprints, dim=0)[0]

        if self.normalize_output:
            phis = self.norm_layer(phis)

        return phis

    @property
    def output_dim(self):
        return self.output_size


if __name__ == '__main__':
    data = ["COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCC1CCN(C)CC1",
           "O[C@@H]1[C@@H](O)[C@@H](Cc2ccccc2)N(C\C=C\c2cn[nH]c2)C(=O)N(C\C=C\c2cn[nH]c2)[C@@H]1Cc1ccccc1",
           "O[C@@H]1[C@@H](O)[C@@H](Cc2ccccc2)N(CC2CC2)C(=O)N(C\C=C\c2cn[nH]c2)[C@@H]1Cc1ccccc1",
           "OCCCCCCN1[C@H](Cc2ccccc2)[C@H](O)[C@@H](O)[C@@H](Cc2ccccc2)N(CC2CC2)C1=O",
           "OCCCCCN1[C@H](Cc2ccccc2)[C@H](O)[C@@H](O)[C@@H](Cc2ccccc2)N(CC2CC2)C1=O",
           "CCCCN1[C@H](Cc2ccccc2)[C@H](O)[C@@H](O)[C@@H](Cc2ccccc2)N(CC2CC2)C1=O",
           "O[C@@H]1[C@@H](O)[C@@H](Cc2ccccc2)N(CC2CCC2)C(=O)N(CC2CCC2)[C@@H]1Cc1ccccc1",
           "OCCCCCN1[C@H](Cc2ccccc2)[C@H](O)[C@@H](O)[C@@H](Cc2ccccc2)N(CCCCCO)C1=O",
           "CCCCN1[C@H](Cc2ccccc2)[C@H](O)[C@@H](O)[C@@H](Cc2ccccc2)N(CCCC)C1=O"]

    transformer = GraphCnnFeaturesExtractor()
    data = transformer(data)

    model = GraphCnnFeaturesExtractor(vocab_size=transformer.vocab_size, embedding_size=20,
                                      kernel_sizes=[64, 128, 258])
    res = model(data)
    print(res.size())

