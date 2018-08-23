from sklearn.model_selection import ParameterGrid
from few_shot_regression.utils_data.preprocessing_movielens import NB_FEATURES

datasets = ['movielens']
examples_per_episode = [10]   # [5, 10, 15, 20, 30, 40, 50, 65, 80, 90]
nb_examples_per_epoch = int(2.5e4)
batch_size = 64
max_episodes = None     # int(1e5)
features_extractor_params_cnn = dict(
    input_size=[NB_FEATURES],
    hidden_sizes=[[256 for _ in range(3)]],
    normalize_features=[False])

expt_settings = dict(
    dataset_name=datasets,
    arch=['fc'],
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
