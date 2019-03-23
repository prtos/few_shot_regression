import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class DeepSetEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(DeepSetEncoder, self).__init__()

        layers = []
        in_dim, out_dim = input_dim, hidden_dim
        for i in range(1, num_layers + 1):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim, out_dim = out_dim, out_dim
        self.phi_net = nn.Sequential(*layers)

        layers = []
        in_dim, out_dim = hidden_dim * 2, hidden_dim
        for i in range(1, num_layers + 1):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim, out_dim = out_dim, out_dim
        self.rho_net = nn.Sequential(*layers)

    def _forward(self, x):
        phis_x = self.phi_net(x)
        sum_x = torch.sum(phis_x, dim=0, keepdim=True)
        max_x, _ = torch.max(phis_x, dim=0, keepdim=True)
        maxsum_x = torch.cat([sum_x, max_x], dim=1)
        res = self.rho_net(maxsum_x).squeeze(0)
        return res

    def forward(self, x):
        return torch.stack([self._forward(x_i) for x_i in x], dim=0)


class Set2SetEncoder(torch.nn.Module):
    r"""
    Set2Set global pooling operator from the `"Order Matters: Sequence to sequence for sets"
    <https://arxiv.org/abs/1511.06391>`_ paper. This pooling layer performs the following operation

    .. math::
        \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

        \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

        \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

    where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
    the dimensionality as the input.

    Arguments
    ---------
        input_dim: int
            Size of each input sample.
        hidden_dim: int, optional
            the dim of set representation which corresponds to the input dim of the LSTM in Set2Set.
            This is typically the sum of the input dim and the lstm output dim. If not provided, it will be set to :obj:`input_dim*2`
        steps: int, optional
            Number of iterations :math:`T`. If not provided, the number of nodes will be used.
        num_layers : int, optional
            Number of recurrent layers (e.g., :obj:`num_layers=2` would mean stacking two LSTMs together)
            (Default, value = 1)
    """

    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(Set2SetEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_lstm_dim = input_dim * 2
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.hidden_lstm_dim, self.input_dim, num_layers=num_layers, batch_first=True)
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(self.hidden_lstm_dim, self.hidden_dim)

    def forward(self, x):
        r"""
        Applies the pooling on input tensor x

        Arguments
        ----------
            x: torch.FloatTensor
                Input tensor of size (B, N, D)

        Returns
        -------
            x: `torch.FloatTensor`
                Tensor resulting from the  set2set pooling operation.
        """
        batch_size, n, _ = x.shape

        h = (x.new_zeros((self.num_layers, batch_size, self.input_dim)),
             x.new_zeros((self.num_layers, batch_size, self.input_dim)))

        q_star = x.new_zeros(batch_size, 1, self.hidden_lstm_dim)

        for i in range(n):
            # q: batch_size x 1 x input_dim
            q, h = self.lstm(q_star, h)
            # e: batch_size x n x 1
            e = torch.matmul(x, q.transpose(1, 2))
            a = self.softmax(e)
            r = torch.sum(a * x, dim=1, keepdim=True)
            q_star = torch.cat([q, r], dim=-1)

        return self.linear(torch.squeeze(q_star, dim=1))


class AttentionLayer(nn.Module):
    def __init__(self, input_size, value_size, key_size, pooling_function=None):
        # input_size == query_size
        super(AttentionLayer, self).__init__()
        self.query_network = nn.Linear(input_size, key_size)
        self.key_network = nn.Linear(input_size, key_size)
        self.value_network = nn.Linear(input_size, value_size)
        self.norm_layer = nn.LayerNorm(value_size)
        self.pooling_function = pooling_function

    def forward(self, query, key=None, value=None):
        if key is None and value is None:
            key = query
            value = query
        assert query.dim() == 3

        query = self.query_network(query)
        key = self.key_network(key)
        value = self.value_network(value)
        attention_matrix = torch.bmm(query, key.transpose(1, 2))
        attention_matrix = attention_matrix / math.sqrt(query.size(2))
        attention_matrix = F.softmax(attention_matrix, dim=2)
        res = self.norm_layer(torch.bmm(attention_matrix, value))

        if self.pooling_function == 'max':
            res = torch.max(res, dim=1)[0]
        elif self.pooling_function == 'mean':
            res = torch.mean(res, dim=1)

        return res


class MultiHeadAttentionLayer(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, num_heads, input_size, value_size, key_size, pooling_function=None, dropout=0.1, residual=True):
        super(MultiHeadAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size

        self.query_network = nn.Linear(input_size, num_heads * key_size)
        self.key_network = nn.Linear(input_size, num_heads * key_size)
        self.value_network = nn.Linear(input_size, num_heads * value_size)

        self.norm_layer = nn.LayerNorm(input_size)
        self.out_layer = nn.Linear(num_heads * value_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.pooling_function = pooling_function
        self.residual = residual

    def forward(self, queries, keys=None, values=None):
        if keys is None and values is None:
            keys = queries
            values = queries
        key_size, value_size, n_head = self.key_size, self.value_size, self.num_heads

        sz_b, len_q, _ = queries.size()
        sz_b, len_k, _ = keys.size()
        sz_b, len_v, _ = values.size()

        residual = queries

        queries = self.query_network(queries).view(sz_b, len_q, n_head, key_size)
        keys = self.key_network(keys).view(sz_b, len_k, n_head, key_size)
        values = self.value_network(values).view(sz_b, len_v, n_head, value_size)

        queries = queries.permute(2, 0, 1, 3).contiguous().view(-1, len_q, key_size)  # (n*b) x lq x dk
        keys = keys.permute(2, 0, 1, 3).contiguous().view(-1, len_k, key_size)  # (n*b) x lk x dk
        values = values.permute(2, 0, 1, 3).contiguous().view(-1, len_v, value_size)  # (n*b) x lv x dv

        attentions = torch.bmm(queries, keys.transpose(1, 2))
        attentions = attentions / math.sqrt(self.key_size)
        attentions = F.softmax(attentions, dim=2)
        outputs = torch.bmm(attentions, values)

        outputs = outputs.view(n_head, sz_b, len_q, value_size)
        outputs = outputs.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        res = self.dropout(self.out_layer(outputs))
        res = self.norm_layer(res + residual) if self.residual else self.norm_layer(res)

        if self.pooling_function == 'max':
            res = torch.max(res, dim=1)[0]
        elif self.pooling_function == 'mean':
            res = torch.mean(res, dim=1)

        return res


class StandardAttentionEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=20, num_layers=1):
        super(StandardAttentionEncoder, self).__init__()
        in_dim, out_dim = input_dim, hidden_dim
        layers = []
        for i in range(1, num_layers + 1):
            pf = None if i != num_layers else 'mean'
            layers.append(AttentionLayer(in_dim, out_dim, in_dim, pooling_function=pf))
            if i != num_layers:
                layers.append(nn.ReLU())
            in_dim, out_dim = out_dim, out_dim
        self._output_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultiHeadAttentionEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=20, num_layers=1, residual=True):
        super(MultiHeadAttentionEncoder, self).__init__()
        in_dim, out_dim = input_dim, hidden_dim
        layers = []
        for i in range(1, num_layers + 1):
            pf = None if i != num_layers else 'mean'
            layers.append(MultiHeadAttentionLayer(8, in_dim, value_size=out_dim, key_size=in_dim, pooling_function=pf, residual=residual))
            if i != num_layers:
                layers.append(nn.ReLU())
        self._output_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RelationNetEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=20, num_layers=1):
        super(RelationNetEncoder, self).__init__()
        in_dim, out_dim = input_dim * 2, hidden_dim
        self.net = DeepSetEncoder(in_dim, out_dim)

    def _forward(self, x):
        n = x.shape[0]
        i, j = torch.tril(torch.ones(n, n, dtype=x.dtype) - torch.eye(n, n, dtype=x.dtype)).nonzero().t()
        x_ = torch.cat((x[i], x[j]), dim=1).unsqueeze(0)
        return self.net(x_).squeeze(0)

    def forward(self, x):
        return torch.stack([self._forward(x_i) for x_i in x], dim=0)


class RegDatasetEncoder(torch.nn.Module):
    def __init__(self, arch, input_dim, target_dim, num_layers=1, hidden_dim=20):
        super(RegDatasetEncoder, self).__init__()
        in_dim, out_dim = (input_dim + target_dim), hidden_dim

        arch = arch.lower()
        if arch in ['set', 'set2set', 's2s']:
            self.net = Set2SetEncoder(in_dim, out_dim, num_layers=num_layers)
            self._output_dim = out_dim
        elif arch in ['standard_attention', 'simple_attention', 's_att', 'att']:
            self.net = StandardAttentionEncoder(in_dim, out_dim, num_layers=num_layers)
            self._output_dim = out_dim
        elif arch in ['multihead_attention', 'mh_att']:
            self.net = MultiHeadAttentionEncoder(in_dim, out_dim, num_layers=num_layers, residual=True)
            self._output_dim = in_dim
        elif arch in ['relation', 'relation_net', 'rn']:
            self.net = RelationNetEncoder(in_dim, out_dim, num_layers=num_layers)
            self._output_dim = out_dim
        elif arch in ['deepset', 'ds']:
            self.net = DeepSetEncoder(in_dim, out_dim, num_layers=num_layers)
            self._output_dim = out_dim
        else:
            raise Exception('arch is undefined')

    def extract_and_pool(self, inputs, targets):
        idx = torch.argsort(targets[:, 0])
        x = torch.cat((inputs[idx], targets[idx]), dim=1).unsqueeze(0)
        return self.net(x).squeeze(0)

    def forward(self, batch_of_set_x_y):
        # not tensorized to accomodate variable dataset sizes in one batch
        features = [self.extract_and_pool(x, y) for (x, y) in batch_of_set_x_y]
        return torch.stack(features, dim=0)

    @property
    def output_dim(self):
        return self._output_dim
