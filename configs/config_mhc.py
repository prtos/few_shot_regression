from sklearn.model_selection import ParameterGrid
from few_shot_regression.utils.feature_extraction.transformers import AMINO_ACID_ALPHABET
datasets = ['mhc']
examples_per_episode = [10]   # [5, 10, 15, 20, 30, 40, 50, 65, 80, 90]
nb_examples_per_epoch = int(2.5e4)
batch_size = 64
max_episodes = None     # int(2.5e4)
features_extractor_params_cnn = dict(
    vocab_size=[1+len(AMINO_ACID_ALPHABET)],
    embedding_size=[20],
    cnn_sizes=[[256 for _ in range(3)]],
    kernel_size=[2],
    dilatation_rate=[2],
    pooling_len=[1],
    lr=[1e-3],
    normalize_features=[False])

expt_settings = dict(
    dataset_name=datasets,
    arch=['cnn'],
    fold=range(14),
    k_per_episode=examples_per_episode,
)

grid_krr = dict(
    fit_params=list(ParameterGrid(dict(
        l2=[0.1],
        **features_extractor_params_cnn
    ))),
    **expt_settings
)


grid_hyperkrr = dict(
    algo=['hyperkrr'],
    fit_params=list(ParameterGrid(dict(
        l2=[0.1],
        use_addictive_loss=[True, False],
        **features_extractor_params_cnn
    ))),
    **expt_settings
)
grid_fskrr = dict(algo=['fskrr'], **grid_krr)
grid_metakrr = dict(algo=['metakrr'], **grid_krr)
grid_multitask = dict(algo=['multitask'], **grid_krr)


grid_maml = dict(
    algo=['maml'],
    fit_params=list(ParameterGrid(dict(
        lr_learner=[0.01],
        n_epochs_learner=[1, 3],
        **features_extractor_params_cnn
    ))),
    **expt_settings
)

grid_mann = dict(
    algo=['mann'],
    fit_params=list(ParameterGrid(dict(
        memory_shape=[(128, 40), (64, 40)],
        controller_size=[200, 100],
        **features_extractor_params_cnn
    ))),
    **expt_settings
)

grid_snail = dict(
    algo=['snail'],
    fit_params=list(ParameterGrid(dict(
        k=examples_per_episode,
        arch=[
            [('att', (64, 32)), ('tc', 128), ('att', (256, 128)), ('tc', 128), ('att', (512, 256))],
            [('att', (64, 32)), ('tc', 128), ('att', (256, 128))]
        ],
        **features_extractor_params_cnn
    ))),
    **expt_settings
)

# grid_krr_lstm = dict(
#     dataset_name=datasets,
#     arch=['lstm'],
#     algo=['krr'],
#     max_examples_per_episode=examples_per_episode,
#     fit_params=list(ParameterGrid(dict(
#         embedding_size=[20],
#         lstm_hidden_size=[32, 64],
#         nb_lstm_layers=[2],
#         bidirectional=[True],
#         normalize_features=[True],
#         unique_l2=[True, False],
#         lr=[1e-3, 1e-4]
#     ))),
#     eval_params=[None]
# )
