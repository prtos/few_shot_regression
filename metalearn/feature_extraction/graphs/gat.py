import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import activation_map, pooling_map
from functools import partial


def message_attn_fn(edges):
    return {"feat": edges.src['feat'], 'attn': edges.src['attn']}


def edge_message_attn_fn(edges):
    return {"efeat": edges.data['he'], 'eattn': edges.data["attn"]}


class GATInit(nn.Module):
    def __init__(self, in_size, out_size, dropout=0., **kwargs):
        super(GATInit, self).__init__()
        self.fc = nn.Linear(in_size, out_size)  # , bias=False)
        self.attn_l = nn.Linear(out_size, 1, bias=False)
        self.attn_r = nn.Linear(out_size, 1, bias=False)
        # you may have to use partial if required
        self.init_fn = kwargs.get("init_fn", partial(
            nn.init.xavier_normal_, gain=1.414))
        self.reset_parameters()
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        if self.init_fn:
            # see paper for initialization scheme
            self.init_fn(self.attn_l.weight)
            self.init_fn(self.attn_r.weight)
            self.init_fn(self.fc.weight)

    def forward(self, X):
        h = X
        if self.dropout:
            h = self.dropout(h)
        feat = self.fc(h)
        attn1 = self.attn_l(feat)
        attn2 = self.attn_r(feat)
        return {'h': h, 'feat': feat, 'self_a': attn1, 'attn': attn2}


class GATReduce(nn.Module):
    def __init__(self, dropout=0.):
        super(GATReduce, self).__init__()
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, nodes):
        # nodes is a batch of node of size B
        # node self attention, not sure why...
        # see https://github.com/PetarV-/GAT/blob/master/utils/layers.py
        self_a = torch.unsqueeze(nodes.data['self_a'], 1)  # shape (B, 1, 1)
        # neigbors contribution (attention)
        attn = nodes.mailbox['attn']  # shape (B, deg, 1)
        feat = nodes.mailbox['feat']  # shape (B, deg, D)
        # attention
        a = self_a + attn  # shape (B, deg, 1)
        e = F.softmax(F.leaky_relu(a), dim=1)
        # attention dropout
        if self.dropout:
            e = self.dropout(e)
        return {'accum': torch.sum(e * feat, dim=1)}  # shape (B, D)


class GATUpdate(nn.Module):
    def __init__(self, block_id, in_size, out_size, activation, b_norm=False, residual=False, **kwargs):
        super(GATUpdate, self).__init__()
        self.block_id = block_id
        self.activation = activation
        self.in_size = in_size
        self.out_size = out_size
        self.init_fn = kwargs.get("init_fn")
        self.residual = residual
        self.res_fc = None
        self.b_norm = None
        if b_norm:
            self.b_norm = nn.BatchNorm1d(self.out_size)
        if self.residual and self.in_size == self.out_size:
            self.res_fc = nn.Linear(in_size, out_size, bias=False)
            if self.init_fn:
                self.init_fn(self.res_fc.weight.data)

    def forward(self, nodes):
        out = nodes.data['accum']
        if self.residual:
            if self.res_fc is not None:
                out = self.res_fc(nodes.data['h']) + out
            else:
                out = out + nodes.data['h']
        if self.b_norm:
            out = self.b_norm(out)
        out = self.activation(out)
        return {'block:{}'.format(self.block_id): out}


class GATLayer(nn.Module):
    """Graph attention (single) layer
    https://arxiv.org/pdf/1710.10903.pdf
    """
    def __init__(self, in_size, block_out_size, attn_block=1, indrop=0., dropout=0., activation="relu", pooling='sum', residual=False, mode="concat", **kwargs):
        super(GATLayer, self).__init__()

        self.in_size = in_size  # this is just for reference. Use graph conv  dim in code
        self.out_size = block_out_size
        self.attn_block = attn_block
        self.mode = mode  # mode can be either avg or concat
        self.residual = residual
        self.activation = activation_map.get(activation, None)
        # on purpose so we can raise an error
        self.pooling = pooling_map[pooling]

        # you may have to use partial if required
        self.init_fn = kwargs.get("init_fn", partial(
            nn.init.xavier_normal_, gain=1.414))

        self.init_attn = nn.ModuleList()
        self.reduce_msg = nn.ModuleList()
        self.finalize_msg = nn.ModuleList()

        for block in range(self.attn_block):
            self.init_attn.append(
                GATInit(self.in_size, self.out_size, indrop, **kwargs))
            self.reduce_msg.append(GATReduce(dropout))
            self.finalize_msg.append(GATUpdate(block, self.in_size, self.out_size, activation=self.activation,
                                               residual=self.residual, b_norm=kwargs.get("b_norm"), init_fn=self.init_fn))

    @property
    def out_features(self):
        if "cat" in self.mode.lower():
            return self.out_size * self.attn_block
        return self.out_size

    def gather(self, G):
        # we need to unpack the graph here
        phis = dgl.sum_nodes(G, 'h')
        #glist = dgl.unbatch(G)
        #phis = torch.squeeze(torch.stack(
        #    [self.pooling(g.ndata["h"].unsqueeze(0)) for g in glist], dim=0), dim=1)
        # then set it back again to normal size
        return phis

    def forward(self, batch_G):
        # note that self loop have been added to all nodes
        G = batch_G
        if not isinstance(G, dgl.BatchedDGLGraph):
            G = dgl.batch(G)
        # why the following line ?
        # to allow another GAT layer, even if that's truly redundant
        h = G.ndata.get("h", G.ndata["hv"])
        # normalization by square root of src degree
        for block in range(self.attn_block):
            G.ndata.update(self.init_attn[block](h))
            G.update_all(message_attn_fn,
                         self.reduce_msg[block], self.finalize_msg[block])

        if "cat" in self.mode:
            h = torch.cat([G.pop_n_repr('block:{}'.format(b_id))
                           for b_id in range(self.attn_block)], dim=1)
        else:
            h = torch.cat([G.pop_n_repr('block:{}'.format(b_id)).unsqueeze(0)
                           for b_id in range(self.attn_block)], dim=0)
            h = h.mean(dim=0)

        G.ndata["h"] = h
        return G, self.gather(G)


class EdgeAttnInit(nn.Module):
    def __init__(self, in_size, out_size, **kwargs):
        super(EdgeAttnInit, self).__init__()
        self.fc = nn.Linear(in_size, out_size)  # , bias=False)
        self.eattn = nn.Linear(out_size, 1, bias=False)
        # you may have to use partial if required
        self.init_fn = kwargs.get("init_fn", partial(
            nn.init.xavier_normal_, gain=1.414))
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_fn:
            # see paper for initialization scheme
            self.init_fn(self.eattn.weight)
            self.init_fn(self.fc.weight)

    def forward(self, X):
        h = X
        feat = self.fc(h)  # esize, D
        attn = self.eattn(feat)
        return {'he': feat, 'attn': attn}


class EdgeReduce(GATReduce):

    def forward(self, nodes):
        # nodes is a batch of node of size B
        # neigbors contribution (attention)
        edge_attn = nodes.mailbox['eattn']  # shape (B, deg, 1)
        feat = nodes.mailbox['efeat']  # shape (B, deg, D)
        # attention
        e = F.softmax(edge_attn, dim=1)  # no leaky relu in the paper
        # see https://arxiv.org/pdf/1802.04944v1.pdf
        # attention dropout
        if self.dropout:
            e = self.dropout(e)
        # taking the weight sum, none of that concat bullshit
        return {'accum': torch.sum(e * feat, dim=1)}  # shape (B, D),


class EdgeAttnLayer(GATLayer):
    # In contrast to GAT, there is no point of having several round of edge attention,
    # in my opinion
    def __init__(self, in_size, block_out_size, attn_block=1, dropout=0., activation="relu", mode="concat", **kwargs):
        super(EdgeAttnLayer, self).__init__(in_size, block_out_size,
                                            attn_block, dropout=0., activation="relu", mode=mode, **kwargs)
        self.init_attn = nn.ModuleList()
        self.reduce_msg = nn.ModuleList()
        self.finalize_msg = nn.ModuleList()

        for block in range(self.attn_block):
            self.init_attn.append(EdgeAttnInit(
                self.in_size, self.out_size, **kwargs))
            self.reduce_msg.append(EdgeReduce(dropout))
            self.finalize_msg.append(GATUpdate(
                block, self.in_size, self.out_size, self.activation, residual=False, init_fn=self.init_fn))

    def gather(self, G):
        # lazy programming 101
        glist = dgl.unbatch(G)
        phis = torch.squeeze(torch.stack(
            [self.pooling(g.ndata["h"].unsqueeze(0)) for g in glist], dim=0), dim=1)
        return phis

    def forward(self, batch_G):
        # self loop should not be here for molecular graphs
        G = batch_G
        if not isinstance(G, dgl.BatchedDGLGraph):
            G = dgl.batch(G)
        h = G.edata["he"]
        # normalization by square root of src degree
        for block in range(self.attn_block):
            G.edata.update(self.init_attn[block](h))
            G.update_all(edge_message_attn_fn,
                         self.reduce_msg[block], self.finalize_msg[block])

        if "cat" in self.mode:
            h = torch.cat([G.pop_n_repr('block:{}'.format(b_id))
                           for b_id in range(self.attn_block)], dim=1)
        else:
            h = torch.cat([G.pop_n_repr('block:{}'.format(b_id)).unsqueeze(0)
                           for b_id in range(self.attn_block)], dim=0)
            h = h.mean(dim=0)

        G.ndata["h"] = h  # add edge feature to nodes
        return G, self.gather(G)
