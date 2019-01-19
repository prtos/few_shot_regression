from sklearn.model_selection import ParameterGrid

shared_params = dict(
    dataset_name=['easytoy'],
    dataset_params=[dict(max_examples_per_episode=10, batch_size=32, max_tasks=None)],
    fit_params=[dict(n_epochs=50, steps_per_epoch=500)],
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


hp_grid_perspectron = dict(
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

hp_grid_deepprior = dict(
    model_name=['deep_prior'],
    model_params=list(ParameterGrid(dict(
        lr=[0.001],
        beta_kl=[0.1, 1],
        input_dim=[1],
        fusion_layer_size=[128],
        fusion_nb_layer=[6],
        feature_extractor_params=features_extractor_params,
        task_encoder_params=list(ParameterGrid(dict(
            target_dim=[1],
            latent_space_dim=[64, 128],
            pooling_mode=['mean'],
        ))),
    ))),
    **shared_params
)

grid_mars = dict(
    model_name=['mars'],
    model_params=list(ParameterGrid(dict(
        l2=[0.1],
        lr=[0.001],
        n_estimators=[4],
        cooling_factor=[1],
        feature_extractor_params=features_extractor_params,
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

grid_metagp_sk = dict(
    model_name=['metagp_sk'],
    model_params=list(ParameterGrid(dict(
        l2=[None],
        lr=[0.001],  # [0.001, 0.01],
        feature_extractor_params=features_extractor_params,
    ))),
    **shared_params
)

grid_metarf = dict(
    model_name=['metarf'],
    model_params=list(ParameterGrid(dict(
        l2=[0.1],
        lr=[0.001],
        n_estimators_train=[10],
        n_estimators_test=[100],
        feature_extractor_params=features_extractor_params,
    ))),
    **shared_params
)

grid_metaboost = dict(
    model_name=['metaboost'],
    model_params=list(ParameterGrid(dict(
        l2=[0.1],
        lr=[0.001],
        n_estimators=[100],
        feature_extractor_params=features_extractor_params,
    ))),
    **shared_params
)

model_params_comm_mk = dict(
    lr=[0.001],  # [0.001],
    beta_kl=[1], # [0, 1e-3, 1e-2],
    use_task_var=[True],
    feature_extractor_params=features_extractor_params,
    conditioner_params=[
        # dict(arch='EW'),
        # *list(ParameterGrid(dict(arch=['TF'], nb_matrix=[25]))),
        *list(ParameterGrid(dict(arch=['LF'], hidden_size=[64], nb_layers=[3])))
    ]
)

model_params_mk = [
    # dict(
    #     use_data_encoder=[True],
    #     task_descr_extractor_params=task_descr_extractor_params + [None],
    #     **model_params_comm_mk
    # ),
    # dict(
    #     use_data_encoder=[False],
    #     task_descr_extractor_params=task_descr_extractor_params,
    #     **model_params_comm_mk
    # ),
    dict(
        l2=[0.1],
        use_data_encoder=[True],
        use_hloss=[True],
        task_descr_extractor_params=[None],
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

if __name__ == '__main__':
    import json
    fname = '../../aws/local_test/test_dir/input/config/hyperparameters.json'
    with open(fname, 'w') as f:
        json.dump(ParameterGrid(grid_metakrr_mk)[0], f)
