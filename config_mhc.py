from sklearn.model_selection import ParameterGrid

expts_directory = 'expt_results/mhc'
datasets = ['mhcpana', 'mhcpanb']
examples_per_episode = [10]
features_extractor_params_cnn = dict(
    embedding_size=[20],
    cnn_sizes=[[25, 25, 25]],
    kernel_size=[2],
    dilatation_rate=[2],
    pooling_len=[1],
    normalize_features=[True])

grid_krr_cnn = dict(
    dataset_name=datasets,
    arch=['cnn'],
    algo=['krr'],
    max_examples_per_episode=examples_per_episode,
    fit_params=list(ParameterGrid(dict(
        unique_l2=[True, False],
        lr=[1e-3, 1e-4],
        **features_extractor_params_cnn
    ))),
    eval_params=[None]
)

grid_maml_cnn = dict(
    dataset_name=datasets,
    arch=['cnn'],
    algo=['maml'],
    max_examples_per_episode=examples_per_episode,
    fit_params=list(ParameterGrid(dict(
        lr=[1e-3, 1e-4],
        lr_learner=[0.01],
        n_epochs_learner=[1, 2, 3, 5],
        **features_extractor_params_cnn
    ))),
    eval_params=[None]
)

grid_pretrain_cnn = dict(
    dataset_name=datasets,
    arch=['cnn'],
    algo=['pretrain'],
    max_examples_per_episode=examples_per_episode,
    fit_params=list(ParameterGrid(dict(
        lr=[1e-3],
        **features_extractor_params_cnn
    ))),
    eval_params=[dict(
        n_epochs=[5, 10, 20],
        lr=[1e-4, 1e-3]
    )]
)

grid_mann_cnn = dict(
    dataset_name=datasets,
    arch=['cnn'],
    algo=['mann'],
    max_examples_per_episode=examples_per_episode,
    fit_params=list(ParameterGrid(dict(
        memory_shape=[(128, 40), (64, 40)],
        controller_size=[200, 100],
        lr=[1e-3],
        **features_extractor_params_cnn
    ))),
    eval_params=[None]
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
#
# grid_maml_lstm = dict(
#     dataset_name=datasets,
#     arch=['lstm'],
#     algo=['maml'],
#     max_examples_per_episode=examples_per_episode,
#     fit_params=list(ParameterGrid(dict(
#         embedding_size=[20],
#         lstm_hidden_size=[32, 64],
#         nb_lstm_layers=[2],
#         bidirectional=[True],
#         normalize_features=[True],
#         lr=[1e-3, 1e-4],
#         lr_learner=[0.01],
#         n_epochs_learner=[1, 2, 3, 5]
#     ))),
#     eval_params=[None]
# )
#
# grid_pretrain_lstm = dict(
#     dataset_name=datasets,
#     arch=['lstm'],
#     algo=['pretrain'],
#     max_examples_per_episode=examples_per_episode,
#     fit_params=list(ParameterGrid(dict(
#         embedding_size=[20],
#         lstm_hidden_size=[32, 64],
#         nb_lstm_layers=[2],
#         bidirectional=[True],
#         normalize_features=[True],
#         lr=[1e-3]
#     ))),
#     eval_params=[dict(
#         n_epochs=[1, 5, 10],
#         lr=[1e-4, 1e-3]
#     )]
# )
#
# grid_mann_lstm = dict(
#     dataset_name=datasets,
#     arch=['lstm'],
#     algo=['mann'],
#     max_examples_per_episode=examples_per_episode,
#     fit_params=list(ParameterGrid(dict(
#         embedding_size=[20],
#         lstm_hidden_size=[32, 64],
#         nb_lstm_layers=[2],
#         bidirectional=[True],
#         normalize_features=[True],
#         memory_shape=[(128, 40)],
#         controller_size=[200],
#         lr=[1e-3]
#     ))),
#     eval_params=[None]
# )