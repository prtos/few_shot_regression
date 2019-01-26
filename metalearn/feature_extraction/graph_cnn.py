from torch.nn import ReLU, Embedding, ModuleList, BatchNorm1d
from metalearn.feature_extraction.common_modules import *
import torch.nn.functional as F
from rdkit import Chem
from itertools import zip_longest
import numpy as np
from metalearn.feature_extraction.common_modules import *
from metalearn.feature_extraction.graphs import GATLayer, BattagliaNMP, DuvenaudNMP, DTNN, GCNLayer


class GraphCnnFeaturesExtractor(ClonableModule):
    def __init__(self, implementation_name, atom_dim, bond_dim, hidden_size, block_size=3, readout_size=32):
        super(GraphCnnFeaturesExtractor, self).__init__()
        # GAT anf GCN need to be stacked by layers, otherwise
        self.is_module = False
        if implementation_name not in ['bmpn', 'dmpn']:
            self.is_module = True
            if isinstance(hidden_size, int):
                hidden_size = [hidden_size]
            self.net = ModuleList()
            for hdim in hidden_size:
                if implementation_name == "attn":
                    self.net.append(
                        GATLayer(atom_dim, hdim, block_size, dropout=0.1, mode="avg"))
                else:
                    self.net.append(GCNLayer(atom_dim, hdim))
                atom_dim = hdim
        elif implementation_name == "bmpn":
            self.net = BattagliaNMP(atom_dim, bond_dim, hidden_size)
        elif implementation_name == "dmpn":
            self.net = DuvenaudNMP(
                atom_dim, bond_dim, hidden_size, readout_size)
        # removed the dtnn layers, because it can be it has too many parameters
        # and was actually designed for coulomb matrices
        # elif implementation_name =="dtnn":
        #    self.net = DTNN(atom_dim, bond_dim, hidden_size, readout_size, readout_size*2)
        else:
            raise ValueError(
                "Unknown implementation name : {}".format(implementation_name))

    def forward(self, input_x):
        if not self.is_module:
            return self.net(input_x)[1]
        G = input_x
        for layer in self.net:
            G, h = layer(G)
        return h

    @property
    def output_dim(self):
        nmp = self.net
        if self.is_module:
            nmp = self.net[-1]
        if hasattr(nmp, 'out_features'):  # should always be there though
            return nmp.out_features
        elif hasattr(nmp, 'output_dim'):
            return nmp.output_dim
        else:
            raise Exception(
                'The graph must provide a method to know the output dimension')
