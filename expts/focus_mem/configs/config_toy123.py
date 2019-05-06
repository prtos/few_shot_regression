from .common_settings import *

shared_params = dict(
    dataset_name=['toy123'],
    dataset_params=[dict(max_examples_per_episode=10, batch_size=32, max_tasks=max_tasks)],
    fit_params=[dict(n_epochs=n_epochs, steps_per_epoch=steps_per_epoch)],
)

features_extractor_params = list(ParameterGrid(dict(
    arch=['fc'],
    input_size=[1],
    hidden_sizes=[[128] * 2],
    normalize_features=[False])))

task_descr_extractor_params = list(ParameterGrid(dict(
    arch=['fc'],
    input_size=[8],
    hidden_sizes=[[16] * 2],
    normalize_features=[False])))

dataset_encoder_params = list(ParameterGrid(dict(
    target_dim=[1],
    nb_layers=[2],
    latent_space_dim=[20],
    pooling_mode=['mean'])))

common_model_params = dict(
    l2=[0.1],
    lr=[0.001],
    kernel=['linear'],
    hp_mode=['f'],  # ['f', 'l', 'cv'],
    feature_extractor_params=features_extractor_params,
)

metakrr_sk = dict(
    model_name=['metakrr_sk'],
    model_params=list(ParameterGrid(common_model_params)),
    **shared_params
)

# fusion_params = list(ParameterGrid(dict(nb_layers=[2], residual=[False])))

mk_model_params = list(ParameterGrid(
    [
        dict(conditioner_mode=['film', 'gated', 'bilinear', 'sum', 'fusion'],
             conditioner_params=[dict(nb_layers=2, residual=False)],
             condition_on=['train', 'descr', 'both'],
             task_descr_extractor_params=task_descr_extractor_params,
             dataset_encoder_params=dataset_encoder_params,
             regularize_dataencoder=[False],
             ** common_model_params),
        dict(conditioner_mode=['film', 'gated', 'bilinear', 'sum', 'fusion'],
             conditioner_params=[dict(nb_layers=2, residual=False)],
             condition_on=['train', 'descr', 'both'],
             task_descr_extractor_params=task_descr_extractor_params,
             dataset_encoder_params=dataset_encoder_params,
             regularize_dataencoder=[True],
             dataenc_beta=[1.0, 0.1, 0.01],
             ** common_model_params),
    ]
))

metakrr_mk = dict(
    model_name=['metakrr_mk'],
    model_params=mk_model_params,
    **shared_params
)

maml = dict(
    model_name=['maml'],
    model_params=list(ParameterGrid(dict(
        lr_learner=[0.01],
        n_epochs_learner=[1, 3],
        feature_extractor_params=features_extractor_params,
    ))),
    **shared_params
)


mann = dict(
    model_name=['mann'],
    model_params=list(ParameterGrid(dict(
        memory_shape=[(128, 40), (64, 40)],
        controller_size=[200, 100],
        feature_extractor_params=features_extractor_params,
    ))),
    **shared_params
)
