import torch
from torch.nn import LSTM, Linear, Sequential, Tanh, Sigmoid
from torch.nn.functional import normalize,  mse_loss, cosine_similarity, softmax, pad
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.optim import Adam
from .base import Model, MetaLearnerRegression
from .modules import ClonableModule, LstmBasedRegressor


class LearnerWithMemory(torch.nn.Module):
    def __init__(self, input_transformer: ClonableModule, controller, controller_to_key,
                 output_layer, gamma, memory, y_last):
        super(LearnerWithMemory, self).__init__()
        self.input_transformer = input_transformer
        self.controller = controller
        self.controller_to_key = controller_to_key
        self.gamma = gamma
        self.memory = memory
        self.output_layer = output_layer
        self.y_last = y_last

    def forward(self, x):
        xs = self.input_transformer(x)
        xys = torch.cat((xs, self.y_last.expand(xs.size(0), 1)), dim=1)
        hiddens, _ = self.controller(xys.unsqueeze(0))
        hiddens = hiddens.view(hiddens.size()[1:])
        keys = self.controller_to_key(hiddens)
        wrs = softmax(torch.mm(normalize(keys), normalize(self.memory).t()), dim=1)
        rs = torch.mm(wrs, self.memory)
        o = torch.cat((hiddens, rs), dim=1)
        y_pred = self.output_layer(o)
        return y_pred


class MANNNetwork(torch.nn.Module):

    def __init__(self, input_transformer: ClonableModule, memory_shape=(128, 40),
                 controller_size=200, nb_reads=1):
        """
        In the constructor we instantiate an lstm module
        """
        super(MANNNetwork, self).__init__()
        self.input_transformer = input_transformer
        self.memory_shape = memory_shape
        self.controller_size = controller_size
        self.nb_reads = nb_reads

        self.controller = LSTM(input_transformer.output_dim+1, self.controller_size)
        self.controller_to_key = Sequential(Linear(self.controller_size, memory_shape[1]), Tanh())
        self.controller_to_sigma = Sequential(Linear(self.controller_size, 1), Sigmoid())
        self.gamma = Parameter(torch.FloatTensor([0.95]), requires_grad=True)
        self.output_layer = Linear(memory_shape[1] + controller_size, 1)

    def get_init_values(self, ):
        temp_1 = torch.ones(self.memory_shape)
        temp_2 = torch.zeros(self.memory_shape[0])
        temp_2[0] = 1
        if torch.cuda.is_available():
            temp_1 = temp_1.cuda()
            temp_2 = temp_2.cuda()

        # temp = torch.zeros(self.controller_size)
        # controller_c_0, controller_h_0 = Variable(temp.clone()), Variable(temp.clone())
        # read_vector_0 = Variable(torch.zeros(self.memory_shape[1]))

        wr_0 = Variable(temp_2.clone(), requires_grad=False)
        wu_0 = Variable(temp_2.clone(), requires_grad=False)
        return Memory_0, wr_0, wu_0

    def __forward(self, episode):
        x_train, y_train = episode['Dtrain']
        xs = self.input_transformer(x_train)
        y_train_shifted, y_last = pad(y_train, (0, 0, 1, 0))[:-1], y_train[-1]
        xys = torch.cat((xs, y_train_shifted), dim=1)
        hiddens, _ = self.controller(xys.unsqueeze(0))
        hiddens = hiddens[0]
        # print(hiddens.size())
        keys = self.controller_to_key(hiddens)
        sigmas = self.controller_to_sigma(hiddens)

        M_tm1, wr_tm1, wu_tm1 = self.get_init_values()
        for t in range(keys.shape[0]):
            h_t, k_t, sigma_t = hiddens[t], keys[t], sigmas[t]
            m = torch.max(wu_tm1)

            wlu_tm1 = (wu_tm1 <= m).float()
            ww_t = (sigma_t * wr_tm1) + ((1 - sigma_t) * wlu_tm1)
            M_t = M_tm1 + torch.ger(ww_t, k_t)
            # print(torch.mm(normalize(M_t), normalize(k_t.unsqueeze(1))).size())
            wr_t = softmax(torch.mm(normalize(M_t), normalize(k_t.unsqueeze(1))).view(-1), dim=0)
            wu_t = (self.gamma * wu_tm1) + wr_t + ww_t

            M_tm1, wr_tm1, wu_tm1 = M_t, wr_t, wu_t
            # r_t = torch.sum(wr_t * M_t, dim=0)

        learner = LearnerWithMemory(self.input_transformer, self.controller, self.controller_to_key,
                                 self.output_layer, self.gamma, M_tm1, y_last)
        return learner(episode['Dtest'])

    def forward(self, episodes):
        return [self.__forward(episode) for episode in episodes]


class MANN(MetaLearnerRegression):
    def __init__(self, learner_network: ClonableModule, loss=mse_loss, lr=0.001,
                 memory_shape=(128, 40), controller_size=200, nb_reads=1):
        super(MANN, self).__init__()
        self.lr = lr
        self.learner_network = learner_network
        self.loss = loss
        self.network = MANNNetwork(learner_network, memory_shape, controller_size, nb_reads)

        if torch.cuda.is_available() and not isinstance(learner_network, LstmBasedRegressor):
            self.network.cuda()

        optimizer = Adam(self.network.parameters(), lr=self.lr)
        self.model = Model(self.network, optimizer, self.metaloss)

    def fit(self, metatrain, metavalid, n_epochs=100, steps_per_epoch=100,
            log_filename=None, checkpoint_filename=None):
        if isinstance(self.learner_network, LstmBasedRegressor):
            metavalid.use_available_gpu = False
            metatrain.use_available_gpu = False
        return super(MANN, self).fit(metatrain, metavalid, n_epochs, steps_per_epoch, log_filename, checkpoint_filename)

