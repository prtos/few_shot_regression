import dgl
import torch
import torch.nn as nn
from .common import activation_map, pooling_map


class GCNLayer(nn.Module):
    """Graph Convolution layer
    Normalization of the graph with addition of self loop is required
    """

    def __init__(self, in_size, out_size, dropout=0., activation="relu", pooling='sum', b_norm=True, bias=True, **kwargs):
        super(GCNLayer, self).__init__()

        self.in_size = in_size  # this is just for reference. Use graph conv  dim in code
        self.out_size = out_size
        self.fc = nn.Linear(self.in_size, self.out_size, bias=bias)
        self.dropout = None
        self.b_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        if b_norm:
            self.b_norm = nn.BatchNorm1d(self.out_size)
        self.activation = activation_map.get(activation, None)
        # on purpose so we can raise an error
        self.pooling = pooling_map[pooling]
        # you will have to use partial if required
        self.init_fn = kwargs.get("init_fn", None)
        self.reset_parameters()

    @property
    def out_features(self):
        return self.out_size

    def reset_parameters(self):
        if self.init_fn:
            self.init_fn(self.fc.weight)

    def gather(self, G):
        # we need to unpack the graph here
        phis = dgl.sum_nodes(G, 'h')
        #glist = dgl.unbatch(G)
        #phis = torch.squeeze(torch.stack(
        #    [self.pooling(g.ndata["h"].unsqueeze(0)) for g in glist], dim=0), dim=1)
        # then set it back again to normal size
        phis = phis.view(-1, self.out_size)
        return phis

    def forward(self, batch_G):
        # note that self loop have been added to all node
        # before end to enable the sparse mat multiplication trick
        G = batch_G
        if not isinstance(G, dgl.BatchedDGLGraph):
            G = dgl.batch(G)
        h = G.ndata.get("h", G.ndata["hv"])
        # normalization by square root of src degree
        h = h * G.ndata.get('norm', 1)
        G.ndata['h'] = h
        G.update_all(dgl.function.copy_src(src='h', out='m'),
                     dgl.function.sum(msg='m', out='h'))
        h = G.ndata.pop('h')
        # renormalize by square root of dst node degree
        h = h * G.ndata.get('norm', 1)
        # we are doing w\sum{h} +b
        # and not \sum{hw +b}
        h = self.fc(h)
        # run activation before dropout
        if self.activation:
            h = self.activation(h)
        if self.dropout:
            h = self.dropout(h)
        if self.b_norm:
            h = self.b_norm(h)
        G.ndata["h"] = h
        return G, self.gather(G)
