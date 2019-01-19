from sklearn.model_selection import ParameterGrid
from metalearn.feature_extraction.transformers import AMINO_ACID_ALPHABET

shared_params = dict(
    dataset_name=['tox21'],
    dataset_params=ParameterGrid(dict(max_examples_per_episode=[10], 
                    batch_size=[32], fold=range(14), max_tasks=[None])),
    fit_params=[dict(n_epochs=50, steps_per_epoch=500)],
)

features_extractor_params = list(ParameterGrid(dict(
    vocab_size=[1+len(AMINO_ACID_ALPHABET)],
    embedding_size=[20],
    cnn_sizes=[[256 for _ in range(3)]],
    kernel_size=[2],
    dilatation_rate=[2],
    pooling_len=[1],
    normalize_features=[False])))

grid_metakrr_sk = dict(
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

grid_maml = dict(
    model_name=['maml'],
    model_params=list(ParameterGrid(dict(
        lr_learner=[0.01],
        n_epochs_learner=[1, 3],
        feature_extractor_params=features_extractor_params,
    ))),
    **shared_params
)

grid_mann = dict(
    model_name=['mann'],
    model_params=list(ParameterGrid(dict(
        memory_shape=[(128, 40), (64, 40)],
        controller_size=[200, 100],
        feature_extractor_params=features_extractor_params,
    ))),
    **shared_params
)