from sklearn.model_selection import ParameterGrid


expts_directory = 'expts_results/bdb'
datasets = ['bindingdb']
examples_per_episode = [10]
features_extractor_params_cnn = dict(
    embedding_size=[40],
    cnn_sizes=[[256, 256, 256]],
    kernel_size=[2],
    dilatation_rate=[2],
    pooling_len=[1],
    normalize_features=[True, False])

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
        lr=[1e-2],
        lr_learner=[0.01],
        n_epochs_learner=[1, 3, 5, 10],
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
        lr=[1e-3, 1e-4],
        **features_extractor_params_cnn
    ))),
    eval_params=[dict(
        n_epochs=[5, 10, 20],
        lr=[1e-3]
    )]
)

grid_mann_cnn = dict(
    dataset_name=datasets,
    arch=['cnn'],
    algo=['mann'],
    max_examples_per_episode=examples_per_episode,
    fit_params=list(ParameterGrid(dict(
        memory_shape=[(128, 40)],
        controller_size=[64, 128],
        lr=[1e-3, 1e-4],
        **features_extractor_params_cnn
    ))),
    eval_params=[None]
)