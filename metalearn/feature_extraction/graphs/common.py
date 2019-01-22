import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter


def params_getter(param_dict, prefix):
    """Filter a parameter dict to keep only a list of relevant parameter dict
    Args
    ----------
    param_dict: dict

    prefix: str or tuple
        A string or a tuple that contains the required prefix(es) to select parameters of interest

    Returns
    -------
    filtered_param_dict: dict
        Dict of (param key, param value) after filtering, and with the prefix removed
    """
    if not (isinstance(prefix, str) or isinstance(prefix, tuple)):
        raise ValueError(
            "Expect a string or a tuple, got {}".format(type(prefix)))
    return dict((pkey.split(prefix, 1)[-1], pval) for pkey, pval in param_dict.items() if pkey.startswith(prefix))


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())
    

class GlobalMaxPool(nn.Module):
    # see stackoverflow
    # https://stats.stackexchange.com/questions/257321/what-is-global-max-pooling-layer-and-what-is-its-advantage-over-maxpooling-layer
    def __init__(self, dim=1):
        super(GlobalMaxPool, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.max(x, dim=self.dim)[0]

class GlobalAvgPool(nn.Module):
    def __init__(self, dim=1):
        super(GlobalAvgPool, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)

class GlobalSumPool(nn.Module):
    def __init__(self, dim=1):
        super(GlobalSumPool, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.sum(x, dim=self.dim)


class Layer(nn.Module):

    def __init__(self, in_size, out_size, dropout=0., activation='relu', b_norm=True, bias=True, **kwargs):
        super(Layer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.linear = nn.Linear(in_size, out_size, bias=bias)
        self.dropout = None
        self.b_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        if b_norm:
            self.b_norm = nn.BatchNorm1d(out_size)
        self.activation = activation_map[activation]
        self.init_fn = kwargs.get("init_fn", None)
        self.reset_parameters()

    @property
    def out_features(self):
        return self.out_size

    def reset_parameters(self):
        if self.init_fn:
            self.init_fn(self.linear.weight)

    def forward(self, x):
        h = self.linear(x)
        h = self.activation(h)
        if self.dropout:
            h = self.dropout(h)
        if self.b_norm:
            h = self.b_norm(h)
        return h


class GRUUpdate(nn.Module):
    """this class is required for the message update in GGNN"""
    def __init__(self, h_size, weight_init_fn=torch.nn.init.xavier_uniform_):
        super(GRUUpdate, self).__init__()
        self.h_size = h_size
        self.W_z = nn.Linear(2 * h_size, h_size)
        self.W_r = nn.Linear(h_size, h_size, bias=False)
        self.U_r = nn.Linear(h_size, h_size)
        self.W_h = nn.Linear(2 * h_size, h_size)
        
        if weight_init_fn:
            weight_init_fn(self.W_z)
            weight_init_fn(self.W_r)
            weight_init_fn(self.U_r)
            weight_init_fn(self.W_h)

    def update_zm(self, node):
        src_x = node.data['src_x']
        s = node.data['s']
        rm = node.data['accum_rm']
        z = torch.sigmoid(self.W_z(torch.cat([src_x, s], 1)))
        m = torch.tanh(self.W_h(torch.cat([src_x, rm], 1)))
        m = (1 - z) * s + z * m
        return {'m': m, 'z': z}

    def update_r(self, node, zm=None):
        dst_x = node.data['dst_x']
        m = node.data['m'] if zm is None else zm['m']
        r_1 = self.W_r(dst_x)
        r_2 = self.U_r(m)
        r = torch.sigmoid(r_1 + r_2)
        return {'r': r, 'rm': r * m}

    def forward(self, node):
        dic = self.update_zm(node)
        dic.update(self.update_r(node, zm=dic))
        return dic

activation_map = {'tanh': nn.Tanh(), 'relu':nn.ReLU(), 'sigmoid':nn.Sigmoid()}
pooling_map = {"max": GlobalMaxPool(), "avg": GlobalAvgPool(), "sum": GlobalSumPool()}
