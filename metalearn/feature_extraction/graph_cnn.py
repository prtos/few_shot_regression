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
        if isinstance(hidden_size, (list, tuple)) and implementation_name not in ['bmpn', 'dmpn']:
            hidden_size = hidden_size[0] 
        if implementation_name == "attn":
            self.net = GATLayer(atom_dim, hidden_size, block_size, dropout=0.1, mode="avg")
        elif implementation_name == "bmpn":
            self.net = BattagliaNMP(atom_dim, bond_dim, hidden_size)
        elif implementation_name == "dmpn":
            self.net = DuvenaudNMP(atom_dim, bond_dim, hidden_size, readout_size)
        # removed the dtnn layers, because it can be it has too many parameters
        # and was actually designed for coulomb matrices
        #elif implementation_name =="dtnn":
        #    self.net = DTNN(atom_dim, bond_dim, hidden_size, readout_size, readout_size*2)
        else:
            self.net = GCNLayer(atom_dim, hidden_size)

    def forward(self, input_x):
        return self.net(input_x)[1]

    @property
    def output_dim(self):
        if hasattr(self.net, 'out_features'): #should always be there though
            return self.net.out_features
        elif hasattr(self.net, 'output_dim'):
            return self.net.output_dim
        else:
            raise Exception('The graph must provide a method to know the output dimension')