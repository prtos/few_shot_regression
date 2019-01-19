from sklearn.model_selection import ParameterGrid
from metalearn.feature_extraction.transformers import SMILES_ALPHABET, MOL_ALPHABET, AMINO_ACID_ALPHABET

shared_params_graph = dict(
    dataset_name=['chembl'],
    dataset_params=[dict(use_graph_for_mol=True, max_examples_per_episode=10, batch_size=32)],
    fit_params=[dict(n_epochs=100, steps_per_epoch=250)],
)

shared_params_smiles = dict(
    dataset_name=['chembl'],
    dataset_params=[dict(use_graph_for_mol=False, max_examples_per_episode=10, batch_size=32)],
    fit_params=[dict(n_epochs=100, steps_per_epoch=250)],
)

features_extractor_params_graph = list(ParameterGrid(dict(
    arch=['gcnn'],
    vocab_size=[1+len(MOL_ALPHABET)],
    embedding_size=[50],
    kernel_sizes=[[512 for _ in range(4)]],
    output_size=[1024],
    normalize_features=[False])))

features_extractor_params_smiles = list(ParameterGrid(dict(
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


f_metakrr_sk = lambda graph: dict(
    model_name=['metakrr_sk'],
    model_params=list(ParameterGrid(dict(
        l2=[0.1],
        lr=[0.001],
        feature_extractor_params=features_extractor_params_graph if graph else features_extractor_params_smiles,
    ))),
    **(shared_params_graph if graph else shared_params_smiles)
)


f_maml = lambda graph: dict(
    model_name=['maml'],
    model_params=list(ParameterGrid(dict(
        lr_learner=[0.01],
        n_epochs_learner=[1, 3],
        feature_extractor_params=features_extractor_params_graph if graph else features_extractor_params_smiles,
    ))),
    **(shared_params_graph if graph else shared_params_smiles)
)


f_mann = lambda graph: dict(
    model_name=['mann'],
    model_params=list(ParameterGrid(dict(
        memory_shape=[(128, 40), (64, 40)],
        controller_size=[200, 100],
        feature_extractor_params=features_extractor_params_graph if graph else features_extractor_params_smiles,
    ))),
    **(shared_params_graph if graph else shared_params_smiles)
)

grid_metakrr_sk = []