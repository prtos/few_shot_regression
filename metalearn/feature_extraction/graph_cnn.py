from torch.nn import ReLU, Embedding, ModuleList, BatchNorm1d
import torch.nn.functional as F
from torch import nn
from rdkit import Chem
from itertools import zip_longest
import numpy as np
from metalearn.feature_extraction.common_modules import ClonableModule
from .gcn_layers import GraphConvLayer


class GraphCnnFeaturesExtractor(ClonableModule):
    def __init__(self, in_size, layer_sizes=[64], 
                 name="GCNN", **kwargs):
        super(GraphCnnFeaturesExtractor, self).__init__()
        self.name = name
        self.in_size = in_size
        self.conv_layers = nn.ModuleList()
        self.pack_batch = True
        self.layer_sizes = layer_sizes

        for ksize in layer_sizes:
            gc_params = {}
            if isinstance(ksize, (tuple, list)) and len(ksize)==2: # so i can customize later
                ksize, gc_params = ksize
            gc = GraphConvLayer(G_size=None, in_size=in_size, 
                        kernel_size=ksize, pack_batch=self.pack_batch, **gc_params)
            self.conv_layers.append(gc) 
            in_size = ksize

    def find_node_per_mol(self, G):
        return [g.shape[0] for g in G]

    def forward(self, input_x):
        G, x = zip(*input_x)
        h = x
        n_per_mol = self.find_node_per_mol(G)
        for i, cv_layer in enumerate(self.conv_layers):
            G, h = cv_layer(G, h)
        # h is batch_size, G_size, kernel_size
        # we sum on the graph dimension before going to the fully connected layers
        h = cv_layer.gather(h, nodes_per_mol=n_per_mol) # h is now batch_size, kernel_size
        return h

    @property
    def output_dim(self):
        res = self.layer_sizes[-1]
        if isinstance(res, int):
            return res
        if isinstance(res, (tuple, list)) and len(res)==2:
            return res[0]
        raise Exception('Impossible to find the size of the output dim')
            
