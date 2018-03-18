from sklearn.model_selection import ParameterGrid


expts_directory = 'expts_results/bdb'
datasets = ['bindingdb']
examples_per_episode = [5, 10, 15]
features_extractor_params_cnn = dict(
    embedding_size=[20],
    cnn_sizes=[[512 for _ in range(4)]],
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
        lr=[1e-3],
        center_kernel=[True, False],
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
        lr=[2e-2],
        lr_learner=[0.01],
        n_epochs_learner=[1],
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
        n_epochs=[5, 10],
        lr=[2e-2]
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
