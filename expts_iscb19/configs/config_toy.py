from sklearn.model_selection import ParameterGrid

shared_params = dict(
    dataset_name=['easytoy'],
    dataset_params=[dict(max_examples_per_episode=10, batch_size=32, max_tasks=None)],
    fit_params=[dict(n_epochs=1, steps_per_epoch=5, max_tasks=100)],
)

features_extractor_params = list(ParameterGrid(dict(
    arch=['fc'],
    input_size=[1],
    hidden_sizes=[[32]*2],
    normalize_features=[False])))

task_descr_extractor_params = list(ParameterGrid(dict(
    arch=['fc'],
    input_size=[13],
    hidden_sizes=[[64] * 2],
    normalize_features=[False])))

metakrr_sk = dict(
    model_name=['metakrr_sk'],
    model_params=list(ParameterGrid(dict(
        l2=[0.1],
        lr=[0.001],
        kernel=['rbf'],
        regularize_phi=[False],
        do_cv=[True],
        feature_extractor_params=features_extractor_params,
    ))),
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
