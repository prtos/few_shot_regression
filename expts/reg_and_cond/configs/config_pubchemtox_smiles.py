from .common_settings import *
from metalearn.feature_extraction.transformers import SMILES_ALPHABET, AMINO_ACID_ALPHABET

shared_params = dict(
    dataset_name=['pubchemtox'],
    dataset_params=[dict(use_graph=True, max_examples_per_episode=10, batch_size=32, max_tasks=max_tasks)],
    fit_params=[dict(n_epochs=n_epochs, steps_per_epoch=steps_per_epoch)],
)

features_extractor_params = list(ParameterGrid(dict(
    arch=['cnn'],
    vocab_size=[1 + len(SMILES_ALPHABET)],
    embedding_size=[20],
    cnn_sizes=[[512 for _ in range(2)]],
    kernel_size=[2],
    dilatation_rate=[2],
    pooling_len=[1],
    use_bn=[False],
    normalize_features=[False]))
)

task_descr_extractor_params = list(ParameterGrid(dict(
    arch=['cnn'],
    vocab_size=[1 + len(AMINO_ACID_ALPHABET)],
    embedding_size=[20],
    cnn_sizes=[[256 for _ in range(2)] + [20]],
    kernel_size=[5],
    dilatation_rate=[2],
    pooling_len=[1],
    use_bn=[False],
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

fusion_params = list(ParameterGrid(dict(nb_layers=[2], residual=[False])))

mk_model_params = list(ParameterGrid(
    [
        dict(conditioner_mode=['film', 'gated'],
             conditioner_params=[None],
             task_descr_extractor_params=task_descr_extractor_params,
             dataset_encoder_params=dataset_encoder_params,
             ** common_model_params),
        dict(conditioner_mode=['film', 'gated'],
             conditioner_params=[None],
             task_descr_extractor_params=task_descr_extractor_params,
             dataset_encoder_params=[None],
             ** common_model_params),
        dict(conditioner_mode=['film', 'gated'],
             conditioner_params=[None],
             task_descr_extractor_params=[None],
             dataset_encoder_params=dataset_encoder_params,
             ** common_model_params),
        dict(conditioner_mode=['fusion'],
             conditioner_params=fusion_params,
             task_descr_extractor_params=task_descr_extractor_params,
             dataset_encoder_params=dataset_encoder_params,
             ** common_model_params),
        dict(conditioner_mode=['fusion'],
             conditioner_params=fusion_params,
             task_descr_extractor_params=task_descr_extractor_params,
             dataset_encoder_params=[None],
             ** common_model_params),
        dict(conditioner_mode=['fusion'],
             conditioner_params=fusion_params,
             task_descr_extractor_params=[None],
             dataset_encoder_params=dataset_encoder_params,
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
        n_epochs_learner=[1],
        feature_extractor_params=features_extractor_params
    ))),
    **shared_params
)

mann = dict(
    model_name=['mann'],
    model_params=list(ParameterGrid(dict(
        memory_shape=[(64, 40)],
        controller_size=[100],
        feature_extractor_params=features_extractor_params
    ))),
    **shared_params
)

s_copy = copy.deepcopy(shared_params)
s_copy['dataset_params'][0].update(dict(raw_inputs=True))
fingerprint = dict(
    model_name=['fingerprint'],
    model_params=list(ParameterGrid(dict(
        algo=['kr'],
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
