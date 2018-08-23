import torch, math
import torch.nn as nn
import torch.nn.functional as F


class StandardSelfAttention(nn.Module):
    def __init__(self, input_size, output_size, pooling_function=None):
        super(StandardSelfAttention, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.attention_linear = nn.Linear(self.input_size, self.input_size)
        self.output_linear = nn.Linear(self.input_size, self.output_size)
        self.pooling_function = pooling_function

    def forward(self, x):
        assert x.dim() == 3
        assert x.size(2) == self.input_size
        query = x
        key = self.attention_linear(x)
        value = self.output_linear(x)
        key = key.transpose(1, 2)
        attention_matrix = torch.bmm(query, key)
        attention_matrix = attention_matrix / math.sqrt(self.input_size)
        attention_matrix = F.softmax(attention_matrix, dim=2)
        applied_attention = torch.bmm(attention_matrix, value)
        if self.pooling_function is None:
            res = applied_attention
        elif self.pooling_function == 'max':
            res = torch.max(applied_attention, dim=1)[0]
        elif self.pooling_function == 'mean':
            res = torch.mean(applied_attention, dim=1)
        return res

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'input_size=' + str(self.input_size) \
            + ', output_size=' + str(self.output_size) + ')'
