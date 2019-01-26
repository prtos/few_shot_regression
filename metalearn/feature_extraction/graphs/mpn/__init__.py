import dgl
import torch
from torch import nn
from torch.functional import F
from metalearn.feature_extraction.graphs.common import params_getter, Layer, pooling_map, activation_map
from .messages import *
from .updates import *


class NMPLayer(nn.Module):
    def __init__(self, in_size, out_size, pooling="sum", ftdim=None):
        super(NMPLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.ft_size = ftdim or self.in_size
        self.size_matcher = nn.Linear(self.in_size, self.ft_size)
        self.pooling = pooling_map[pooling]
        self.resample = self.in_size != self.ft_size

    @property
    def out_features(self):
        return self.out_size

    def gather(self, G, xkey="h"):
        glist = dgl.unbatch(G)
        phis = torch.squeeze(torch.stack(
            [self.pooling(g.ndata[xkey].unsqueeze(0)) for g in glist], dim=0), dim=1)
        return phis

    def pack_graph(self, glist, nattr="hv", eattr="he"):
        """Pack graph list of dgl graph into one"""
        if isinstance(glist, dgl.BatchedDGLGraph):
            return glist
        return dgl.batch(glist, nattr, eattr)

    def _scale_dim(self, x):
        """Rescale dim into  graph list of dgl graph into one"""
        if self.resample:
            return self.size_matcher(x)
        return x.requires_grad_()

    def _update_nodes(self, G, nattr="hv", xkey="h"):
        # actually, this is pretty useless
        # see: https://docs.dgl.ai/api/python/batch.html
        # because updating hv has no effect on the original graphs
        # but I am keeping it to allow residual connection
        # and I hope it doesn't increase the memory footprint
        return {xkey: self._scale_dim(G.ndata[nattr])}


class LoopyNMP(NMPLayer):
    def __init__(self, atom_dim, bond_dim, hidden_size, depth, pooling="avg"):
        super(LoopyNMP, self).__init__(atom_dim, hidden_size, pooling=pooling)
        self.n_update = depth
        self.W_i = nn.Linear(atom_dim + bond_dim, hidden_size, bias=False)
        self.loopy_bp_updater = LoopyBPUpdate(hidden_size)
        self.gather_updater = GatherUpdate(atom_dim, hidden_size)
        self.n_samples_total = 0
        self.n_nodes_total = 0
        self.n_edges_total = 0
        self.n_passes = 0

    def forward(self, batch_G):
        G = self.pack_graph(batch_G)
        n_samples = G.batch_size

        mol_line_graph = G.line_graph(backtracking=False, shared=True)

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        G = self.run(G, mol_line_graph)
        # TODO: replace with unbatch or readout
        g_repr = self.gather(G)

        self.n_samples_total += n_samples
        self.n_nodes_total += n_nodes
        self.n_edges_total += n_edges
        self.n_passes += 1

        return g_repr

    def run(self, G, mol_line_graph):
        n_nodes = G.number_of_nodes()

        G.apply_edges(
            func=lambda edges: {'src_h': edges.src['hv']},
        )

        e_repr = mol_line_graph.ndata
        bond_features = e_repr['he']
        source_features = e_repr['src_h']

        features = torch.cat([source_features, bond_features], 1)
        msg_input = self.W_i(features)
        mol_line_graph.ndata.update({
            'msg_input': msg_input,
            'msg': F.relu(msg_input),
            'accum_msg': torch.zeros_like(msg_input),
        })
        G.ndata.update({
            'm': bond_features.new(n_nodes, self.out_size).zero_(),
            'h': bond_features.new(n_nodes, self.out_size).zero_(),
        })

        for i in range(self.n_update - 1):
            mol_line_graph.update_all(
                mpn_loopy_bp_msg,
                mpn_loopy_bp_reduce,
                self.loopy_bp_updater,
            )

        G.update_all(
            mpn_gather_msg,
            mpn_gather_reduce,
            self.gather_updater,
        )

        return G


class BattagliaNMP(NMPLayer):
    """Battagli Interaction network, here there is no accumulation, only the last layer matter"""

    def __init__(self, atom_dim, bond_dim, out_dim, x_dim=0, pooling="sum", **kwargs):
        super(BattagliaNMP, self).__init__(atom_dim, out_dim, pooling)
        self.n_update = 1
        if isinstance(out_dim, list):
            self.n_update = len(out_dim)
        else:
            out_dim = [out_dim]
        self.out_size = out_dim[-1]
        self.mpn_msg = nn.ModuleList()
        self.mpn_updt = nn.ModuleList()
        msg_kwargs = params_getter(kwargs, ("msg__", "m__"))
        updt_kwargs = params_getter(
            kwargs, ("updater__", "u__", "upd__", "updt__"))
        for hdim in out_dim:
            self.mpn_msg.append(BattagliaMsg(
                atom_dim+atom_dim+bond_dim, hdim, **msg_kwargs))
            self.mpn_updt.append(BattagliaUpdate(
                atom_dim+hdim, hdim, x_dim, **updt_kwargs))
            atom_dim = hdim

    def gather(self, G, xkey="h"):
        # we need to unpack the graph here
        phis = dgl.sum_nodes(G, xkey)
        return phis

    def forward(self, batch_G):
        G = self.pack_graph(batch_G)
        n_nodes = G.number_of_nodes()
        G.ndata.update(self._update_nodes(G, "hv", "h"))
        for updt in range(self.n_update):
            G.update_all(send_msg, self.mpn_msg[updt], self.mpn_updt[updt])
        return G, self.gather(G)


class DuvenaudNMP(NMPLayer):
    """Duvenaud NMP. For some weird reasons, the original paper does not describe the same thing
    as in Gilmer and al. or not even in the provided code written in autograd
    So, I am following the gilmer and al. specs"""

    def __init__(self, atom_dim, bond_dim, out_dim, readout_dim, max_degree=5, **kwargs):
        super(DuvenaudNMP, self).__init__(atom_dim, readout_dim, pooling="sum")
        self.n_update = 1
        if isinstance(out_dim, list):
            self.n_update = len(out_dim)
        else:
            out_dim = [out_dim]
        self.mpn_updt = nn.ModuleList()
        self.mpn_msg = nn.ModuleList()
        msg_kwargs = params_getter(kwargs, ("msg__", "m__"))
        updt_kwargs = params_getter(
            kwargs, ("updater__", "u__", "upd__", "updt__"))
        for hdim in out_dim:
            self.mpn_msg.append(DuvenaudMsg(
                atom_dim+bond_dim, hdim, max_degree, **msg_kwargs))
            self.mpn_updt.append(DuvenaudUpdate(
                hdim, readout_dim, **updt_kwargs))
            atom_dim = hdim
        self.W_o = nn.Linear(self.in_size, readout_dim, bias=False)


    def gather(self, G, xkey="rd"):
        # we need to unpack the graph here
        phis = dgl.sum_nodes(G, xkey)
        return phis


    def forward(self, batch_G):
        G = self.pack_graph(batch_G)
        # n_nodes = G.number_of_nodes()
        # print(G.in_degrees().unique(), "**** ")
        # print(n_nodes, "****\n")
        G.ndata.update(self._update_nodes(G, "hv", "h"))
        G.ndata.update({
            'rd': F.softmax(self.W_o(G.ndata["h"]), dim=1)
        })
        for updt in range(self.n_update):
            G.update_all(send_msg, self.mpn_msg[updt], self.mpn_updt[updt])
        return G, self.gather(G, "rd")


class DTNN(NMPLayer):
    """Deep Tensor Neural Networks
    https://www.nature.com/articles/ncomms13890
    """

    def __init__(self, atom_dim, bond_dim, msg_hidden_dim, out_dim, ft_dim=None, **kwargs):
        super(DTNN, self).__init__(
            atom_dim, out_dim, pooling="sum", ftdim=ft_dim)
        if not isinstance(msg_hidden_dim, (list, tuple)):
            msg_hidden_dim = [msg_hidden_dim]
        self.n_update = len(msg_hidden_dim)
        msg_kwargs = params_getter(kwargs, ("msg__", "m__"))
        self.mpn_msg = nn.ModuleList()
        for hdim in msg_hidden_dim:
            self.mpn_msg.append(
                DTNNmsg(self.ft_size, bond_dim, hdim, **msg_kwargs))
        self.readout = kwargs.get("net", Layer(
            self.ft_size, self.out_size, **kwargs))

    def gather(self, G):
        # lazy programming 101
        glist = dgl.unbatch(G)
        phis = torch.squeeze(torch.stack([self.pooling(self.readout(
            g.ndata["h"]).unsqueeze(0)) for g in glist], dim=0), dim=1)
        return phis

    def forward(self, batch_G):
        G = self.pack_graph(batch_G)
        G.ndata.update(self._update_nodes(G, "hv", "h"))
        for updt in range(self.n_update):
            G.update_all(send_nn_msg, self.mpn_msg[updt])
            G.ndata.update({
                "h": G.ndata["h"]+G.ndata["m"]
            })
        return G, self.gather(G)


class NeuralGraphFingerprint(NMPLayer):
    """This is the direct implementation of duvenaud neuralfingerprint as described in their paper
    Nothing about bond data and all that.
    https://arxiv.org/pdf/1509.09292.pdf
    """

    def __init__(self, atom_dim, layers_dim, fp_dim=512, pooling="sum", bias=False, **kwargs):
        """Normalization of the graph by adding self_loop is required"""
        super(NeuralGraphFingerprint, self).__init__(
            atom_dim, fp_dim, pooling="sum")
        self.n_update = 1
        if isinstance(layers_dim, list):
            self.n_update = len(layers_dim)
        else:
            layers_dim = [layers_dim]
        self.layers = nn.ModuleList()
        for hdim in layers_dim:
            self.layers.append(
                NGFUpdate(atom_dim, hdim, fp_dim, sp_bias=bias, **kwargs))
            atom_dim = hdim

    def forward(self, batch_G):
        G = self.pack_graph(batch_G)
        n_nodes = G.number_of_nodes()
        G.ndata.update(self._update_nodes(G, "hv", "h"))
        G.ndata.update({
            'rd': G.ndata["h"].new(n_nodes, self.out_size).zero_().requires_grad_()
        })
        for updt in range(self.n_update):
            G.update_all(ng_fp_msg, ng_fp_reduce, self.layers[updt])

        return G, self.gather(G, "rd")


class GGNN(NMPLayer):
    def __init__(self, in_dim, msg_dim, out_dim, edge_types, n_update=1, **kwargs):
        super(GGNN, self).__init__(in_dim, out_dim)
        self.n_update = n_update
        self.out_size = out_dim
        self.edge_types = edge_types
        self.pooling = pooling_map[kwargs.get('pooling', 'sum')]
        self.mpn_msg = nn.ModuleList()
        self.mpn_updt = nn.GRU(msg_dim, in_dim)
        self.init_fn = kwargs.get("init_fn")
        self.reset_parameters()
        self.inet = kwargs.get('inet', Layer(
            self.in_dim*2, self.out_size, activation="sigmoid", bias=False, b_norm=False))
        self.jnet = kwargs.get('jnet', Layer(
            self.in_dim*2, self.out_size, activation="tanh"))
        self.activation = activation_map[kwargs.get('activation', 'tanh')]
        for hdim in out_dim:
            self.mpn_msg.append(GGNNMsg(in_dim, msg_dim, self.edge_types))

    def reset_parameters(self):
        if self.init_fn:
            for name in self.mpn_updt.named_parameters():
                if 'weight' in name[0]:
                    self.init_fn(name[1])

    def update(self, nodes):
        # receive a batch of nodes
        # nodes:
        h = nodes.data['h'].unsqueeze(0)  # for num_layer dim
        m = nodes.data['m'].unsqueeze(0)  # for input dim
        h_new, _ = self.mpn_updt(m, h)
        return {'h': h_new.squeeze(0)}

    def readout(self, h_0, h_T):
        # as described in original paper
        # https://arxiv.org/pdf/1511.05493.pdf
        # and not the gilmer one
        out_i = self.inet(torch.cat([h_T, h_0], dim=1))
        out_j = self.jnet(torch.cat([h_T, h_0], dim=1))
        return out_i*out_j

    def gather(self, G, xkey="rd"):
        glist = dgl.unbatch(G)
        phis = torch.squeeze(torch.stack(
            [self.pooling(g.ndata[xkey].unsqueeze(0)) for g in glist], dim=0), dim=1)
        return phis

    def forward(self, batch_G):
        G = self.pack_graph(batch_G)
        G.ndata.update(self._update_nodes(G, "hv", "h"))
        h_0 = G.ndata['h']
        for updt in range(self.n_update):
            G.update_all(send_nn_msg, self.mpn_msg[updt], self.update)
        h_T = G.ndata['h']
        G.ndata["rd"] = self.readout(h_0, h_T)
        return G, self.gather(G)


class WeaveGather(nn.Module):
    def __init__(self, in_dim=128, activation='tanh', **kwargs):
        super(WeaveGather, self).__init__()
        self.in_dim = in_dim
        self.activation = activation_map[activation]
        self.memberships = [(-1.645, 0.283), (-1.080, 0.170), (-0.739, 0.134),
                            (-0.468, 0.118), (-0.228, 0.114), (0., 0.114),
                            (0.228, 0.114), (0.468, 0.118), (0.739, 0.134),
                            (1.080, 0.170), (1.645, 0.283)]
        self.mdim = len(self.memberships)
        self.linear = nn.Linear(self.in_dim*self.mdim, self.in_dim)

    def run(self, x):
        dist = [torch.distributions.normal.Normal(mu, sigma) for (mu, sigma) in self.memberships]
        dist_max = [dist[i].log_prob(self.memberships[i][0]) for i in range(self.mdim)]
        outputs = torch.cat([torch.exp(dist[i].log_prob(x) - dist_max[i]) for i in range(self.mdim)], dim=1)
        outputs = outputs / torch.sum(outputs, dim=1)
        outputs = outputs.view(-1, self.in_dim * self.mdim)
        return outputs
    
    def forward(self, x):
        outputs = self.linear(x)
        outputs = self.activation(x)
        return outputs


class WeaveNet(NMPLayer):
    """Molecular Graph Convolutions, Kearnes et al. (2016)
    See https://arxiv.org/abs/1603.00856
    only one update is done. 
    This is not the Gilmer et al. description, but the original paper.
    see also: https://goo.gl/i6K72W for deepchem take on this
    """

    def __init__(self, in_dim, msg_in_dim, h_dim, msg_out_dim, W_dim=None, readout_dim=128, bias=True, b_norm=False, **kwargs):
        super(WeaveNet, self).__init__(in_dim, readout_dim)
        self.h_dim = h_dim
        self.W_dim = W_dim
        self.msg_out_dim = msg_out_dim
        if not self.W_dim or (isinstance(self.W_dim, (list, tuple)) and len(self.W_dim) < 5):
            # in the paper they use 50, 50, 50, 50
            self.W_dim = [in_dim, msg_in_dim, in_dim, msg_in_dim]
        self.AA = nn.Linear(self.in_dim, self.W_dim[0], bias=bias)
        self.PA = nn.Linear(msg_in_dim, self.W_dim[1], bias=bias)
        self.A = nn.Linear(
            self.W_dim[0]+self.W_dim[1], self.h_dim, bias=bias)
        # edge params
        self.AP = nn.Linear(in_dim*2, self.W_dim[2], bias=bias)
        self.PP = nn.Linear(msg_in_dim, self.W_dim[3], bias=bias)
        self.P = nn.Linear(self.W_dim[2]+self.W_dim[3], msg_out_dim, bias=bias)

        self.activation = activation_map[kwargs.get('activation', 'relu')]
        self.msg_fn = dgl.function.copy_edge('he', 'msg')
        self.bond_update = kwargs.get('bond_update', True)
        self.readout_layer = Layer(self.h_dim, self.out_dim, activation='tanh', b_norm=True)
        self.gather_met = kwargs.get('pooling', 'gaussian')
        self.gaussian_layer = WeaveGather(self.out_dim)

    def get_out_dim(self):
        return self.h_dim, self.msg_out_dim

    def reduce_fn(self, nodes):
        # nodes: BATCH, DEG, FT
        msg = nodes.mailbox['msg']
        hpa = self.PA(msg)
        hpa = self.activation(hpa)
        hpa = torch.sum(hpa, dim=1)  # sum on degree dim
        return {'m': hpa}

    def node_update(self, nodes):
        # nodes:  Batch, Features
        hpa = nodes.data['m']
        haa = self.AA(nodes.data['h'])
        haa = self.activation(haa)
        ha = self.AA(torch.cat([haa, hpa], dim=1))
        h = self.activation(ha)
        return {'h': h}

    def edge_update(self, edges):
        # edges: Batch, Features
        hv = edges.src['h']
        hw = edges.dst['h']
        evw = edges.data['he']
        # is this dumb to check both?
        # yes !
        hap_i = self.AP(torch.cat([hv, hw], dim=1))
        hap_i = self.activation(hap_i)
        hap_j = self.AP(torch.cat([hw, hv], dim=1))
        hap_j = self.activation(hap_j)

        hpp = self.PP(evw)
        hpp = self.activation(hpp)

        hp = self.P(torch.cat([hap_i + hap_j, hpp], dim=1))
        hp = self.activation(hp)
        return {'he': hp}

    def gather(self, G, xkey='rd'):
        if self.gather_met == 'gaussian':
            G.ndata.update({xkey: self.gaussian_layer.run(G.ndata[xkey])})
            return self.gaussian_layer(dgl.sum_nodes(G, xkey))
        else:
            return super(WeaveNet, self).gather(G, xkey)

    def forward(self, batch_G):
        G = self.pack_graph(batch_G)
        G.ndata.update(self._update_nodes(G, "hv", "h"))
        G.update_all(self.msg_fn, self.reduce_fn)
        if self.bond_update:
            G.apply_edges(self.edge_update)
        G.apply_nodes(self.node_update)
        G.ndata['rd'] = self.readout_layer(G.ndata['h'])
        return G, self.gather(G)
