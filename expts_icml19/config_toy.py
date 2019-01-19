from sklearn.model_selection import ParameterGrid

shared_params = dict(
    dataset_name=['easytoy'],
    dataset_params=[dict(max_examples_per_episode=10, batch_size=32, max_tasks=None)],
    fit_params=[dict(n_epochs=100, steps_per_epoch=500)],
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


grid_perspectron = dict(
    model_name=['perspectron'],
    model_params=list(ParameterGrid(dict(
        lr=[0.001],
        input_dim=[1],
        target_dim=[1],
        pooling_mode=['mean'],
        is_latent_discrete=[False],   # , False],
        latent_space_dim=[25, 64],
        feature_extractor_params=features_extractor_params,
    ))),
    **shared_params
)

grid_deep_prior = dict(
    model_name=['deep_prior'],
    model_params=list(ParameterGrid(dict(
        lr=[0.001],
        beta_kl=[0.1],
        input_dim=[1],
        fusion_layer_size=[128],
        fusion_nb_layer=[6],
        cotraining=[True, False],
        pretraining=[True, False],
        feature_extractor_params=features_extractor_params,
        task_encoder_params=list(ParameterGrid(dict(
            target_dim=[1],
            latent_space_dim=[128],
            pooling_mode=['mean'],
        ))),
    ))),
    **shared_params
)

grid_metakrr_sk = dict(
    model_name=['metakrr_sk'],
    model_params=list(ParameterGrid(dict(
        l2=[0.1],
        lr=[0.001],
        kernel=['linear'],
        regularize_w_pairs=[False],
        regularize_phi=[False],
        do_cv=[True],
        feature_extractor_params=features_extractor_params,
    ))),
    **shared_params
)
if __name__ == '__main__':
    import json
    fname = 'config.json'
    with open(fname, 'w') as f:
        json.dump(grid_deep_prior, f, indent=2)
