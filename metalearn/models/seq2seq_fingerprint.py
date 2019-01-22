import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from pytoune.framework import Model
from .base import MetaLearnerRegression, FeaturesExtractorFactory, MetaNetwork
from .krr import KrrLearner, KrrLearnerCV
from .utils import reset_BN_stats, to_unit

SOS_token = 0
EOS_token = 1

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class SeqtoseqFPNetwork(MetaNetwork):
    def __init__(self, vocab_size, hidden_size, teacher_forcing_ratio, dropout_p=0.1):
        super(SeqtoseqFPNetwork, self).__init__()
        self.encoder = EncoderRNN(vocab_size, hidden_size)
        self.decoder = AttnDecoderRNN(hidden_size, vocab_size, dropout_p=dropout_p)
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, sequences):

        inputs = [ep['Dtrain'][0] for ep in episodes]
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if np.random.rand() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

    



class MetaKrrSingleKernelLearner(MetaLearnerRegression):
    def __init__(self, *args, optimizer='adam', lr=0.001, weight_decay=0.0, **kwargs):
        network = MetaKrrSingleKernelNetwork(*args, **kwargs)
        super(MetaKrrSingleKernelLearner, self).__init__(network, optimizer, lr, weight_decay)

    def _compute_aux_return_loss(self, y_preds, y_tests):
        res = dict()
        loss = torch.mean(torch.stack([mse_loss(y_pred, y_test) 
                    for y_pred, y_test in zip(y_preds, y_tests)]))

        res.update(dict(mse=loss))
        if self.model.training:
            x = torch.mean(torch.cat(self.model.phis_norms))
            res.update(dict(phis_norm=x))
            if self.model.regularize_phi:
                loss = loss + x

        return loss, res

if __name__ == '__main__':
    pass
