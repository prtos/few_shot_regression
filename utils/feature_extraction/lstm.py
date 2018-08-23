from torch.nn import LSTM, Linear, Sequential, Module
from torch.nn.utils.rnn import pack_padded_sequence
from few_shot_regression.utils.feature_extraction.common_modules import *


class LstmFeaturesExtractor(Module):
    def __init__(self, vocab_size, embedding_size, lstm_hidden_size, nb_lstm_layers,
                 bidirectional=True, normalize_features=True):
        super(LstmFeaturesExtractor, self).__init__()
        self.params = locals()
        del self.params['__class__']
        del self.params['self']
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lstm_hidden_size = lstm_hidden_size
        self.nb_lstm_layers = nb_lstm_layers
        self.batch_first = True
        self.concat_lstms = True
        self.normalize_output = normalize_features
        self.bidirectional = bidirectional
        self.embedding = EmbeddingViaLinear(vocab_size, self.embedding_size)
        self.lstm = LSTM(input_size=self.embedding_size,
                                        num_layers=self.nb_lstm_layers,
                                        hidden_size=self.lstm_hidden_size,
                                        bidirectional=self.bidirectional,
                                        batch_first=self.batch_first)
        if self.concat_lstms:
            self.od = lstm_hidden_size * nb_lstm_layers * (2 if self.bidirectional else 1)
        else:
            self.od = lstm_hidden_size

        if self.normalize_output:
            self.norm_layer = UnitNormLayer()

    @property
    def output_dim(self):
        return self.od

    def forward(self, x):
        x_ = x.view(x.size(0), -1)
        lengths = (x_ > 0).long().sum(1)
        lengths, perm_idx = lengths.sort(0, descending=True)
        _, rev_perm_idx = perm_idx.sort(0)

        x = x[perm_idx]
        batch_size = x.size(0)
        # h_0 = Variable(torch.zeros((x_train.size(0), self.nb_layers * self.num_directions, self.hidden_size)))
        embedding = self.embedding(x)
        packed_x_train = pack_padded_sequence(embedding, lengths.data.cpu().numpy(), batch_first=self.batch_first)
        packed_output, (hidden, _) = self.lstm(packed_x_train)
        # output, _ = pad_packed_sequence(packed_output, batch_first=self.batch_first)
        if self.concat_lstms:
            hidden = hidden.transpose(0, 1).contiguous()
            phis = hidden.view(batch_size, -1)
        else:
            phis = hidden[-1]

        if self.normalize_output:
            phis = self.norm_layer(phis)

        return phis[rev_perm_idx]

    def clone(self):
        model = LstmFeaturesExtractor(**self.params)
        if next(self.parameters()).is_cuda:
            model = model.cuda()
        return model




if __name__ == '__main__':
    pass