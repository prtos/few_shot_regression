import torch
from torch import nn
from torch.functional import F
from metalearn.feature_extraction.graphs.common import Layer, activation_map

class LoopyBPUpdate(nn.Module):
    def __init__(self, hidden_size):
        super(LoopyBPUpdate, self).__init__()
        self.hidden_size = hidden_size

        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, nodes):
        msg_input = nodes.data['msg_input']
        msg_delta = self.W_h(nodes.data['accum_msg'])
        msg = F.relu(msg_input + msg_delta)
        return {'msg': msg}


class GatherUpdate(nn.Module):
    def __init__(self, in_size, hidden_size, activation="relu"):
        super(GatherUpdate, self).__init__()
        self.hidden_size = hidden_size
        self.activation = activation_map[activation]
        self.W_o = nn.Linear(in_size + hidden_size, hidden_size)

    def forward(self, nodes):
        m = nodes.data['m']
        return {
            'h': self.activation(self.W_o(torch.cat([nodes.data['hv'], m], 1))),
        }


class BattagliaUpdate(nn.Module):
    """Battagli Interaction network, here there is no accumulation, only the last layer matter"""
    def __init__(self, in_dim, out_dim, x_dim=0,  **kwargs):
        super(BattagliaUpdate, self).__init__()
        self.in_size = in_dim + x_dim
        self.x_dim = x_dim
        self.out_size = out_dim
        self.net = kwargs.get("net", Layer(
            self.in_size, self.out_size, bias=False, **kwargs))  # change this

    def forward(self, nodes):
        h = nodes.data['h']
        msg = nodes.data['m'] 
        if self.x_dim:
            x = nodes.data["ext"]
            out = torch.cat([h, x, msg], dim=1)
        else:
            out = torch.cat([h, msg], dim=1)
        out = self.net(out)
        return {'h': out}


class DuvenaudUpdate(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(DuvenaudUpdate, self).__init__()
        self.in_size = in_dim
        self.out_size = out_dim
        self.W_t = nn.Linear(self.in_size, self.out_size, bias=False)

    def forward(self, nodes):
        h = nodes.data['h']  # shape (B, E+D)
        rd = F.softmax(self.W_t(h), dim=1) + nodes.data["rd"]
        return {"rd": rd}


class NGFUpdate(nn.Module):
    def __init__(self, in_dim, out_dim, fp_dim, bias=False, **kwargs):
        super(NGFUpdate, self).__init__()
        self.in_size = in_dim
        self.out_size = out_dim
        self.fp_dim = fp_dim
        self.hidden_net = kwargs.get("net", Layer(
            self.in_size, self.out_size, **kwargs))
        self.fp_layer = nn.Linear(self.out_size, self.fp_dim, bias=bias)

    def forward(self, nodes):
        h = nodes.data['h']  # batch, features
        h = self.hidden_net(h)
        rd = F.softmax(self.fp_layer(h), dim=1) + nodes.data["rd"]
        return {'h': h, "rd": rd}