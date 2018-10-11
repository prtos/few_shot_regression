from sklearn.model_selection import ParameterGrid
from metalearn.feature_extraction.transformers import SMILES_ALPHABET, MOL_ALPHABET, AMINO_ACID_ALPHABET

use_graph_for_mol = False

shared_params = dict(
    dataset_name=['bindingdb'],
    dataset_params=[dict(use_graph_for_mol=use_graph_for_mol, max_examples_per_episode=10, batch_size=32)],
    fit_params=[dict(valid_size=0.25, n_epochs=100, steps_per_epoch=250)],
)

if use_graph_for_mol:
    features_extractor_params = list(ParameterGrid(dict(
        arch=['gcnn'],
        vocab_size=[1+len(MOL_ALPHABET)],
        embedding_size=[50],
        kernel_sizes=[[512 for _ in range(4)]],
        output_size=[1024],
        normalize_features=[False])))
else:
    features_extractor_params = list(ParameterGrid(dict(
        arch=['cnn'],
        vocab_size=[1+len(SMILES_ALPHABET)],
        embedding_size=[20],
        cnn_sizes=[[512 for _ in range(4)]],
        kernel_size=[2],
        dilatation_rate=[2],
        pooling_len=[1],
        use_bn=[False],
        normalize_features=[False])))


task_descr_extractor_params = list(ParameterGrid(dict(
    arch=['cnn'],
    vocab_size=[1 + len(AMINO_ACID_ALPHABET)],
    embedding_size=[20],
    cnn_sizes=[[256 for _ in range(2)]],
    kernel_size=[5],
    dilatation_rate=[2],
    pooling_len=[1],
    use_bn=[False],
    normalize_features=[False])))


grid_metakrr_sk = dict(
    model_name=['metakrr_sk'],
    model_params=list(ParameterGrid(dict(
        l2=[0.1],
        lr=[0.001],
        feature_extractor_params=features_extractor_params,
    ))),
    **shared_params
)

grid_metagp_sk = dict(
    model_name=['metagp_sk'],
    model_params=list(ParameterGrid(dict(
        l2=[0.1, 0.01],
        lr=[0.001],
        feature_extractor_params=features_extractor_params,
    ))),
    **shared_params
)

model_params_comm_mk = dict(
    l2=[0.1],
    lr=[0.001],
    # beta_kl=[1e-3, 1e-2, 1e-1],
    use_task_var=[False],
    feature_extractor_params=features_extractor_params,
    conditioner_params=list(ParameterGrid([
        dict(arch=['ew']),
        # dict(arch=['tf'], nb_matrix=[10, 25]),
        dict(arch=['lf'], hidden_sizes=[[128]])
    ]))
)

model_params_mk = [
    dict(
        use_data_encoder=[True],
        task_descr_extractor_params=task_descr_extractor_params + [None],
        **model_params_comm_mk
    ),
    dict(
        use_data_encoder=[False],
        task_descr_extractor_params=task_descr_extractor_params,
        **model_params_comm_mk
    ),
]

grid_metakrr_mk = dict(
    model_name=['metakrr_mk'],
    model_params=list(ParameterGrid(model_params_mk)),
    **shared_params
)

grid_metagp_mk = dict(
    model_name=['metagp_mk'],
    model_params=list(ParameterGrid(model_params_mk)),
    **shared_params
)

grid_multitask = dict(
    model_name=['mutitask'],
    model_params=list(ParameterGrid(dict(
        l2=[0.1, 0.01],
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


grid_snail = dict(
    model_name=['snail'],
    model_params=list(ParameterGrid(dict(
        k=[10],
        arch=[
            [('att', (64, 32)), ('tc', 128), ('att', (256, 128)), ('tc', 128), ('att', (512, 256))],
            [('att', (64, 32)), ('tc', 128), ('att', (256, 128))]
        ],
        feature_extractor_params=features_extractor_params,
    ))),
    **shared_params
)