from sklearn.model_selection import ParameterGrid


expts_directory = 'expts_results/bdb'
datasets = ['bindingdb']
examples_per_episode = [10]

grid_krr_cnn = dict(
    dataset_name=datasets,
    arch=['cnn'],
    algo=['krr'],
    max_examples_per_episode=examples_per_episode,
    fit_params=list(ParameterGrid(dict(
        embedding_size=[20],
        cnn_sizes=[[256, 256]],
        kernel_size=[2],
        pooling_len=[2, 1],
        normalize_features=[True],
        unique_l2=[True, False],
        lr=[1e-3, 1e-4]
    ))),
    eval_params=[None]
)

grid_maml_cnn = dict(
    dataset_name=datasets,
    arch=['cnn'],
    algo=['maml'],
    max_examples_per_episode=examples_per_episode,
    fit_params=list(ParameterGrid(dict(
        embedding_size=[20],
        cnn_sizes=[[256, 256], [128, 128]],
        kernel_size=[2, 5, 10],
        pooling_len=[2, 1],
        normalize_features=[True],
        lr=[1e-3, 1e-4],
        lr_learner=[0.01],
        n_epochs_learner=[1, 3, 5, 10]
    ))),
    eval_params=[None]
)

grid_pretrain_cnn = dict(
    dataset_name=datasets,
    arch=['cnn'],
    algo=['pretrain'],
    max_examples_per_episode=examples_per_episode,
    fit_params=list(ParameterGrid(dict(
        embedding_size=[20],
        cnn_sizes=[[256, 256], [128, 128]],
        kernel_size=[2, 5, 10],
        pooling_len=[2, 1],
        normalize_features=[True],
        lr=[1e-3, 1e-4],
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
        embedding_size=[20],
        cnn_sizes=[[256, 256], [128, 128]],
        kernel_size=[2, 5, 10],
        pooling_len=[2, 1],
        normalize_features=[True],
        memory_shape=[(128, 40)],
        controller_size=[64, 128],
        lr=[1e-3, 1e-4]
    ))),
    eval_params=[None]
)