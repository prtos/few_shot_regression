import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from .common_modules import activation_map, pooling_map, to_sparse


def pack_graph(batch_G, batch_x, return_sparse=False, fill_missing=0):
    """Pack a batch of graph and atom features into a single graph

    Parameters
    ----------
    batch_G : iterable (torch.LongTensor 2D), of size (n_i, n_i). Sparse tensor allowed
    batch_x: iterable (torch.Tensor 2D) of size (n_i,d_i)
    return_sparse: whether to return a sparse graph
    fill_missing: (int) fill out-of-graph bond
    Note that by filling out-of-graph positions, with anything other than 0, you cannot have a sparse tensor.

    Returns
    -------
    new_batch_G, new_batch_x: (torch.LongTensor 2D, torch.Tensor 2D)
        This tuple represent a new arbitrary graph and the corresponding atom feature matrix.
        new_batch_G has size (N, N), with $N = \sum_i n_i$, while new_batch_x has size (N,d)
    """
    out_x = torch.cat(tuple(batch_x), dim=0)
    n_neigb = out_x.shape[0]
    out_G = batch_G[0].new_zeros((n_neigb, n_neigb))
    cur_ind = 0
    n_per_mol = [] # should return this eventually
    for g in batch_G:
        g_size = g.shape[0] + cur_ind
        n_per_mol.append(g.shape[0])
        if g.is_sparse:
            g = g.to_dense()
        out_G[cur_ind:g_size, cur_ind:g_size] = g
        cur_ind = g_size

    if return_sparse and fill_missing == 0:
        out_G = to_sparse(out_G)
    return out_G, out_x


class GraphConvLayer(nn.Module):

    def __init__(self, G_size=150, in_size=75, kernel_size=64, dropout=0., activation='relu', pooling='sum', pack_batch=False,  **kwargs):
        # Accepted method for pooling are avg and sum
        super(GraphConvLayer, self).__init__()
        self.G_size = G_size
        self.in_size = in_size # this is just for reference. Use graph conv  dim in code
        self.kernel_size = kernel_size
        self.linear = nn.Linear(in_size, self.kernel_size)
        self.dropout = nn.Dropout(p=dropout)
        self.b_norm = nn.BatchNorm1d(kernel_size)
        self.activation = activation_map[activation]
        self.use_sparse = kwargs.get("sparse", True)
        self.pack_batch = pack_batch
        self.pooling = pooling_map[pooling] # on purpose so we can raise an error

    def gather(self, h, nodes_per_mol=None):
        if self.pack_batch:
            if not nodes_per_mol:
                raise ValueError("Expect node_per_mol for packed graph")
            return torch.squeeze(torch.stack([self.pooling(mol_feat)
                    for mol_feat in torch.split(h, nodes_per_mol, dim=1)], dim=0), dim=1)
        return torch.squeeze(self.pooling(h), dim=1)

    def _forward(self, h):
        h = h.view(-1, h.size()[-1]) # reshape to fit linear layer requirements
        h = self.linear(h)
        #print('linear', h.shape)
        h = self.activation(h)
        #print('activation', h.shape)
        h = self.dropout(h)
        #print('dropout', h.shape)
        # still debating if batch normalization should be after activation 
        # and dropout.
        # this seems to be the common order
        h = self.b_norm(h) 
        return h 


    def forward(self, G, x):
        """
        G is batch_size, G_size, G_size
        x is batch_size, G_size, in_size
        """
        # we compute (Gx)w (w from the linear layer)
        # same as G(xw) as seen in other implementation
        G_size = self.G_size
        if not self.pack_batch and isinstance(G, (list, tuple)):
            G = torch.stack(G)
            x = torch.stack(x)
        if self.pack_batch:
            if not isinstance(G, torch.Tensor): 
                G, x = pack_graph(G, x, self.use_sparse)
            if x.shape[0] == 1:
                x = x.squeeze(0)
            G_size = x.shape[0]
            h = torch.mm(G, x).unsqueeze(0) # support sparse here for G
            # we add the batch dimension again
        else: # expect a batch here
            G = G.view(-1, G.shape[-2], G.shape[-1]) # ensure that batch dim is there
            h = x.view(-1, x.shape[-2], x.shape[-1])
            h = torch.bmm(G, h) # batch_size, G_size, in_size
            G_size = h.shape[1]

        h = self._forward(h)
        h = h.view(-1, G_size, self.kernel_size) # then set it back again to normal size
        return G, h 