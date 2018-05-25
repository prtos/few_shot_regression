from sklearn.model_selection import ParameterGrid

datasets = ['bindingdb']
examples_per_episode = [10]
nb_examples_per_epoch = int(2e4)
batch_size = 32
max_episodes = int(1e5)
features_extractor_params_cnn = dict(
    embedding_size=[20],
    cnn_sizes=[[512 for _ in range(4)]],
    kernel_size=[2],
    dilatation_rate=[2],
    pooling_len=[1],
    normalize_features=[False])


expt_settings = dict(
    dataset_name=datasets,
    arch=['cnn'],
    k_per_episode=examples_per_episode,
)

grid_krr = dict(
    fit_params=list(ParameterGrid(dict(
        l2=[0.1],
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