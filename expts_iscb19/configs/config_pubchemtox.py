from sklearn.model_selection import ParameterGrid
from metalearn.feature_extraction.transformers import SMILES_ALPHABET, MOL_ALPHABET, AMINO_ACID_ALPHABET, DGLGraphTransformer

transformer_ = DGLGraphTransformer()
dataset_name = 'pubchemtox'
test = False

if test:
    shared_params_graph = dict(
        dataset_name=[dataset_name],
        dataset_params=[dict(use_graph=True, max_examples_per_episode=10, batch_size=32, max_tasks=100)],
        fit_params=[dict(n_epochs=1, steps_per_epoch=5)],
    )

    shared_params_smiles = dict(
        dataset_name=[dataset_name],
        dataset_params=[dict(use_graph=False, max_examples_per_episode=10, batch_size=32, max_tasks=100)],
        fit_params=[dict(n_epochs=1, steps_per_epoch=5)],
    )
else:
    shared_params_graph = dict(
        dataset_name=[dataset_name],
        dataset_params=[dict(use_graph=True, max_examples_per_episode=10, batch_size=32)],
        fit_params=[dict(n_epochs=100, steps_per_epoch=500)],
    )

    shared_params_smiles = dict(
        dataset_name=[dataset_name],
        dataset_params=[dict(use_graph=False, max_examples_per_episode=10, batch_size=32)],
        fit_params=[dict(n_epochs=100, steps_per_epoch=500)],
    )

features_extractor_params_graph = list(ParameterGrid(dict(
    arch=['gcnn'],
    implementation_name=['attn', 'gcn'], 
    atom_dim=[transformer_.n_atom_feat], 
    bond_dim=[transformer_.n_bond_feat],
    hidden_size=[256], 
    readout_size=[32],)))

features_extractor_params_smiles = list(ParameterGrid(dict(
    arch=['cnn'],
    vocab_size=[1+len(SMILES_ALPHABET)],
    embedding_size=[20],
    cnn_sizes=[[512 for _ in range(2)]],
    kernel_size=[2],
    dilatation_rate=[2],
    pooling_len=[1],
    use_bn=[False],
    normalize_features=[False])))


task_descr_extractor_params = list(ParameterGrid(dict(
    arch=['cnn'],
    vocab_size=[1 + len(AMINO_ACID_ALPHABET)],
    embedding_size=[20],
    cnn_sizes=[[512 for _ in range(2)]],
    kernel_size=[5],
    dilatation_rate=[2],
    pooling_len=[1],
    use_bn=[False],
    normalize_features=[False])))


def f_metakrr_sk(graph): 
    return dict(
        model_name=['metakrr_sk'],
        model_params=list(ParameterGrid(dict(
            l2=[0.1],
            lr=[0.001],
            do_cv=[True, False],
            fixe_hps=[True, False],
            kernel=['linear', 'rbf'],
            feature_extractor_params=features_extractor_params_graph if graph else features_extractor_params_smiles,
        ))),
        **(shared_params_graph if graph else shared_params_smiles)
    )


def f_maml(graph):
    return dict(
        model_name=['maml'],
        model_params=list(ParameterGrid(dict(
            lr_learner=[0.01],
            n_epochs_learner=[1],
            feature_extractor_params=features_extractor_params_graph if graph else features_extractor_params_smiles,
        ))),
        **(shared_params_graph if graph else shared_params_smiles)
    )


def f_mann(graph): 
    return dict(
        model_name=['mann'],
        model_params=list(ParameterGrid(dict(
            memory_shape=[(64, 40)],
            controller_size=[100],
            feature_extractor_params=features_extractor_params_graph if graph else features_extractor_params_smiles,
        ))),
        **(shared_params_graph if graph else shared_params_smiles)
    )

import copy
s_copy = copy.deepcopy(shared_params_graph)
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
    **shared_params_graph
)

if test:
    metakrr_sk = f_metakrr_sk(False)
    maml = f_maml(False)
    mann = f_mann(False)
else:
    metakrr_sk = f_metakrr_sk(False)
    maml = f_maml(False)
    mann = f_mann(False)
    # metakrr_sk = [f_metakrr_sk(False), f_metakrr_sk(False)]
    # maml = [f_maml(False), f_maml(False)]
    # mann = [f_mann(False), f_mann(False)]