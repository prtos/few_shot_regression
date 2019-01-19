import torch
from torch.nn import Linear, Sequential, Hardtanh, Tanh, ReLU, Sigmoid, Parameter, LSTM
from torch.nn.functional import mse_loss, log_softmax, nll_loss
from torch.optim import Adam
from pytoune.framework import Model
from perspectron_eai.base.attention import StandardSelfAttention
from tensorboardX import SummaryWriter
from pytoune.framework.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, \
    BestModelRestore, TensorBoardLogger
from perspectron_eai.base.utils import to_unit, Lstm


class SetFeatureExtractor(torch.nn.Module):
    def __init__(self, input_dim, latent_space_dim, agg_arch='attention', pooling_mode='mean'):
        super(SetFeatureExtractor, self).__init__()
        if agg_arch == 'lstm':
            self.agg = Lstm(input_dim, latent_space_dim, nb_lstm_layers=3,
                            bidirectional=False, pooling_function=pooling_mode)
        elif agg_arch == 'attention':
            self.agg = Sequential(
                StandardSelfAttention(input_dim, latent_space_dim, pooling_function=None),
                Tanh(),
                # StandardSelfAttention(latent_space_dim, latent_space_dim, pooling_function=None),
                StandardSelfAttention(latent_space_dim, latent_space_dim, pooling_function=pooling_mode),
                Tanh()
            )
        else:
            raise Exception('DatasetFeatureExtractor: unhandled architecture!')

    def forward(self, batch_of_set_x_y):
        return self.agg(batch_of_set_x_y)


class MetadataEncoderNetwork(torch.nn.Module):
    def __init__(self, input_dim, latent_space_dim, output_dim, pooling_mode='mean'):
        super(MetadataEncoderNetwork, self).__init__()
        assert pooling_mode in ['mean', 'max']
        self.data_feature_extractor = SetFeatureExtractor(latent_space_dim=latent_space_dim, input_dim=input_dim,
                                                          pooling_mode=None)
                                                          # pooling_mode=pooling_mode)
        self.output_dim = output_dim
        self.out_layer = Sequential(
            Linear(4*latent_space_dim, latent_space_dim),
            Tanh(),
            Linear(latent_space_dim, output_dim)
        )
        self.writer = None
        self.step = 0

    def set_writer(self, writer):
        self.writer = writer

    def forward(self, batch_of_episodes):
        phis_data = self.data_feature_extractor(batch_of_episodes)
        phis_data = torch.cat((torch.max(phis_data, dim=1)[0], torch.min(phis_data, dim=1)[0],
                  torch.mean(phis_data, dim=1), torch.std(phis_data, dim=1)), dim=1)
        out = self.out_layer(phis_data)
        return out


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class MetadataEncoderLearner(Model):

    def __init__(self, *args, lr=0.01, feature_extractor=None, x_dim=1, y_dim=1, network=None, **kwargs):
        super(MetadataEncoderLearner, self).__init__(None, None, None)
        self.feature_extractor = feature_extractor
        input_dim = (x_dim + y_dim) if self.feature_extractor is None else (self.feature_extractor.output_dim + y_dim)
        if network is None:
            self.model = MetadataEncoderNetwork(*args, input_dim=input_dim, **kwargs)
            if torch.cuda.is_available():
                self.model.cuda()
        else:
            self.model = network
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.loss_function = self.__compute_aux_and_return_hloss
        self.train_step = 0

    def __compute_aux_and_return_hloss(self, y_outs, y_true):
        global_loss = mse_loss(y_outs, y_true)
        clf_scores = torch.exp(-((y_outs[:, None] - y_true[None, :])** 2).sum(dim=2))
        classes = torch.arange(clf_scores.size(0), dtype=torch.long)
        if torch.cuda.is_available():
            classes = classes.cuda()
        top_1, top_3 = accuracy(clf_scores, classes, topk=(1, 3))

        # metrics = {'MAPE_dim_'+str(i): to_unit(el)
        #            for i, el in enumerate(torch.abs((y_outs - y_true)/y_true).mean(dim=0))}
        metrics = {}
        metrics.update({'MSE_dim_'+str(i): to_unit(el)
                        for i, el in enumerate(((y_outs - y_true)**2).mean(dim=0))})
        metrics.update(dict(top1_acc=top_1, top3_acc=top_3))
        if self.model.training:
            self.train_step += 1
            if self.writer is not None:
                scalars = dict(mse=to_unit(global_loss), **metrics)
                for k, v in scalars.items():
                    self.writer.add_scalar('MetadataEncoder/'+k, v, self.train_step)
        if torch.isnan(global_loss).any():
            raise Exception(f'{self.__class__.__name__}: Loss goes NaN')
        return global_loss

    def fit(self, metatrain, *args, valid_size=0.25, n_epochs=100, steps_per_epoch=100,
            log_filename=None, checkpoint_filename=None, tboard_folder=None, early_stop=False, **kwargs):
        meta_train, meta_valid = metatrain.train_test_split(valid_size)
        meta_train.train()
        meta_valid.train()
        meta_train, meta_valid = MetadatasetWrapper(meta_train, self.feature_extractor), \
                                 MetadatasetWrapper(meta_valid, self.feature_extractor)
        print("Number of train steps:", len(meta_train))
        print("Number of valid steps:", len(meta_valid))

        callbacks = [ReduceLROnPlateau(patience=3, factor=1/2, min_lr=1e-6),
                     BestModelRestore()]
        if early_stop:
            callbacks += [EarlyStopping(patience=5, verbose=False)]
        if log_filename:
            callbacks += [CSVLogger(log_filename, batch_granularity=False, separator='\t')]
        if checkpoint_filename:
            callbacks += [ModelCheckpoint(checkpoint_filename, monitor='val_loss', save_best_only=True,
                                           temporary_filename=checkpoint_filename+'temp')]

        if tboard_folder is not None:
            self.writer = SummaryWriter(tboard_folder)

        self.fit_generator(meta_train, meta_valid,
                           epochs=n_epochs,
                           steps_per_epoch=steps_per_epoch,
                           validation_steps=None,
                           callbacks=callbacks,
                           verbose=True)
        self.is_fitted = True
        return self

    def evaluate(self, metatest, *args, metrics=[], **kwrags):
        ds = MetadatasetWrapper(metatest, self.feature_extractor)
        return self.evaluate_generator(ds, return_pred=True)

    def load(self, checkpoint_filename):
        self.model.load_weights(checkpoint_filename)
        self.is_fitted = True


class MetadatasetWrapper:
    def __init__(self, ds, feature_extractor):
        self.ds = ds
        self.f_ext = feature_extractor

    def __iter__(self):
        for episodes, _ in self.ds:
            x = torch.stack([torch.cat(((self.f_ext(ep['Dtrain'][0]) if self.f_ext else ep['Dtrain'][0]),
                                        ep['Dtrain'][1]), dim=1) for ep in episodes])
            # x = [ep['Dtrain'] for ep in episodes]
            y = torch.stack([ep['task_descr'] for ep in episodes])
            # print(x.shape)
            yield x, y

    def __len__(self):
        return len(self.ds)


if __name__ == '__main__':
    pass
