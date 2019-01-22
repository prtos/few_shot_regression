import dgl
import torch
from torch import nn
from metalearn.feature_extraction.graphs.common import Layer, pooling_map, activation_map

ng_fp_msg = dgl.function.copy_src(src="h", out="m")
ng_fp_reduce = dgl.function.sum(msg="m", out="h")

mpn_gather_msg = dgl.function.copy_edge(edge='msg', out='msg')
mpn_gather_reduce = dgl.function.sum(msg='msg', out='m')

mpn_loopy_bp_msg = dgl.function.copy_src(src='msg', out='msg')
mpn_loopy_bp_reduce = dgl.function.sum(msg='msg', out='accum_msg')


def send_msg(edges):
    msg = torch.cat([edges.src['h'], edges.data["he"]], dim=1)
    return {"msg": msg}

def send_nn_msg(edges):
    return {"src_h": edges.src['h'], "he": edges.data["he"]}

class BattagliaMsg(nn.Module):
    def __init__(self, in_size, out_size, **kwargs):
        super(BattagliaMsg, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.pooling = pooling_map[kwargs.get("pooling", "sum")]
        self.net = kwargs.get("net", Layer(
            self.in_size, self.out_size, bias=False, **kwargs))  # change this

    def forward(self, nodes):
        msg = nodes.mailbox['msg']  # shape (B, deg, D+E)
        batch_size = msg.shape[0]
        h = nodes.data["h"].unsqueeze(dim=1)  # shape (B, 1, D)
        new_shape = list(h.shape)
        new_shape[1] = msg.shape[1]
        h = h.expand(new_shape)  # shape, B, deg, D
        m = torch.cat([h, msg], dim=2)  # shape (B, deg, D+D+E)
        m = self.net(m.view(-1, self.in_size))
        # sum over all neigh
        return {"m": self.pooling(m.view(batch_size, -1, self.out_size))}


class DuvenaudMsg(nn.Module):
    def __init__(self, in_dim, out_dim, max_degree, **kwargs):
        super(DuvenaudMsg, self).__init__()
        self.in_size = in_dim
        self.out_size = out_dim
        self.nets = nn.ModuleList()
        self.max_degree = max_degree
        param_dict = dict(bias=False, b_norm=False, dropout=False, activation="sigmoid")
        param_dict.update(kwargs)
        for deg in range(max_degree):  # adding one for the unexpected degree
            self.nets.append(kwargs.get("net", Layer(
                self.in_size, self.out_size, **param_dict)))  # change this
        self.failed_net = kwargs.get("net", Layer(
            self.in_size, self.out_size, **param_dict))

    def get_net(self, deg):
        if -self.max_degree <= deg < self.max_degree:
            return self.nets[deg]
        return self.failed_net

    def forward(self, nodes):
        # Remember that DGL use degree bucketing,
        # so the batch of nodes here have all the same in-degree (sweet!)
        msg = nodes.mailbox['msg']  # Batch, Deg, Feats
        deg = int(msg.shape[1])
        out = torch.sum(msg, dim=1) # or should we sum on the message ?
        out = self.get_net(deg)(out)
        return {'h': out}


class DTNNmsg(nn.Module):
    def __init__(self, atom_dim, bond_dim, hidden_dim, activation="tanh", pooling="sum"):
        super(DTNNmsg, self).__init__()
        self.out_size = atom_dim  # the message size should match the feature dim
        self.Wcf = nn.Linear(atom_dim, hidden_dim, bias=True)
        self.Wdf = nn.Linear(bond_dim, hidden_dim, bias=True)
        self.Wfc = nn.Linear(hidden_dim, self.out_size)
        self.pooling = pooling_map[pooling]
        self.activation = activation_map[activation]

    def forward(self, nodes):
        h_w = nodes.mailbox['src_h']  # B, deg, feat
        e_vw = nodes.mailbox['he']  # B, deg, feat
        h_w = self.Wcf(h_w)
        e_vw = self.Wdf(e_vw)
        h = torch.mul(h_w, e_vw)
        h = self.Wfc(h)
        # see equation 6 of paper, pooling is the sum
        return {"m": self.pooling(self.activation(h))}


class GGNNMsg(nn.Module):
    def __init__(self, in_dim, out_dim, edge_type, **kwargs):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_type = edge_type
        param_dict = dict(bias=False, b_norm=False, dropout=False)
        self.A = nn.ModuleList()
        param_dict.update(kwargs)
        for etype in range(edge_type):  # adding one for the unexpected degree
            self.A.append(nn.Linear(self.in_dim, self.out_dim, bias=False))
        self.failed_net = nn.Linear(self.in_dim, self.out_dim, bias=False)

    def get_net(self, elabel):
        for i, e in enumerate(self.edge_type):
            if e == elabel:
                return self.A[i]
        return self.failed_net

    def run(self, msg, elabel):
        # msg: deg, feats
        # elable: deg
        # group all incomming with the same label then 
        # run the linear layer on them
        out = torch.cat([self.get_net(e)(torch.index_select(msg, 0, (elabel!=e).nonzero().squeeze(-1))) for e in torch.unique(elabel)], dim=0)
        return torch.sum(out, dim=0) # sum msg for current node

    def forward(self, nodes):
        # Remember that DGL use degree bucketing,
        # so the batch of nodes here have all the same in-degree (sweet!)
        hw = nodes.mailbox['src_h']  # Batch, Deg, Feats
        elabels = nodes.mailbox['he'].squeeze(-1)  # Batch, Deg, 1
        # get hw by edge types:
        out = torch.cat([self.run(hw[i], elabels[i]) for i in range(nodes.batch_size())], dim=0)
        return {'m': out}
