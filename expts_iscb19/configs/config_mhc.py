from sklearn.model_selection import ParameterGrid
from metalearn.feature_extraction.transformers import AMINO_ACID_ALPHABET

shared_params = dict(
    dataset_name=['mhc'],
    dataset_params=list(ParameterGrid(dict(max_examples_per_episode=[10], 
                    batch_size=[32], fold=range(14), max_tasks=[None]))),
    fit_params=[dict(n_epochs=100, steps_per_epoch=500)],
)

features_extractor_params = list(ParameterGrid(dict(
    arch=['cnn'],
    vocab_size=[1+len(AMINO_ACID_ALPHABET)],
    embedding_size=[20],
    cnn_sizes=[[256 for _ in range(3)]],
    kernel_size=[2],
    dilatation_rate=[2],
    pooling_len=[1],
    normalize_features=[False])))

metakrr_sk = dict(
    model_name=['metakrr_sk'],
    model_params=list(ParameterGrid(dict(
        l2=[0.1],
        lr=[0.001],
        kernel=['rbf', 'linear'],
        fixe_hps=[True, False],
        do_cv=[True, False],
        feature_extractor_params=features_extractor_params,
    ))),
    **shared_params
)

maml = dict(
    model_name=['maml'],
    model_params=list(ParameterGrid(dict(
        lr_learner=[0.01],
        n_epochs_learner=[1],
        feature_extractor_params=features_extractor_params,
    ))),
    **shared_params
)

mann = dict(
    model_name=['mann'],
    model_params=list(ParameterGrid(dict(
        memory_shape=[(64, 40)],
        controller_size=[100],
        feature_extractor_params=features_extractor_params,
    ))),
    **shared_params
)

import copy
s_copy = copy.deepcopy(shared_params)
s_copy['dataset_params'][0].update(dict(raw_inputs=True))
fingerprint = dict(
    model_name=['fingerprint'],
    model_params=list(ParameterGrid(dict(
        algo=['rf'],
        fp=['morgan_circular'],
    ))),
    **s_copy
)

seqtoseq = dict(
    model_name=['seqtoseq'],
    model_params=list(ParameterGrid(dict(
        embedding_dim=[256], 
        encoder_layers=[2], 
        decoder_layers=[2], 
        dropout=[0.1],
    ))),
    **shared_params
)