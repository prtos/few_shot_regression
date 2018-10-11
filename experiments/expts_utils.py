import sys, hashlib, json
import pandas as pd
import numpy as np
from metalearn.datasets import loaders
import metalearn.feature_extraction as extraction
import metalearn.models as models
from collections import OrderedDict, MutableMapping

SAVING_DIR_FORMAT = '{expts_dir}/results_{dataset_name}_{algo}_{arch}'
loader_dict = dict(
    bindingdb=loaders.load_episodic_bindingdb,
    movielens=loaders.load_episodic_movielens,
    mhc=loaders.load_episodic_mhc,
    uci=loaders.load_episodic_uciselection,
    toy=loaders.load_episodic_harmonics,
    easytoy=loaders.load_episodic_easyharmonics)
inner_class_dict = dict(
    tcnn=extraction.TcnnFeaturesExtractor,
    cnn=extraction.Cnn1dFeaturesExtractor,
    lstm=extraction.LstmFeaturesExtractor,
    fc=extraction.FcFeaturesExtractor,
    gcnn=extraction.GraphCnnFeaturesExtractor)
metalearnerclass_dict = dict(
    mann=models.MANN,
    maml=models.MAML,
    snail=models.SNAIL,
    deep_prior=models.DeepPriorLearner,
    multitask=models.MultiTaskLearner,
    metakrr_sk=models.MetaKrrSingleKernelLearner,
    metakrr_mk=models.MetaKrrMultiKernelsLearner,
    metagp_sk=models.MetaGPSingleKernelLearner,
    metagp_mk=models.MetaGPMultiKernelsLearner
)


def params_dict_to_str(d):
    e = [(k, d[k]) for k in sorted(d.keys()) if type(d[k]) != bool]
    base = '_'.join([k for k in sorted(d.keys()) if d[k] is True])
    r = str(e).replace(' ', '').replace(':', '_').replace('\'', '').replace(',', '_').replace('(', '').replace(')', '')
    r = base + '_' + r
    return r


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_outfname_prefix(all_params):
    params_flatten = flatten_dict(all_params)
    base = str(params_flatten)
    uid = hashlib.md5(str(base).encode()).hexdigest()

    readable_name = params_flatten.get('dataset_name', '')
    readable_name += '_' + params_flatten.get('model_name', '')
    for k in ['arch', 'nb_matrix', 'hidden_sizes']:
        readable_name += '_' + str(params_flatten.get('model_params.conditioner_params.'+k, ''))
    temp = params_flatten.get('model_params.beta_kl', "")
    readable_name += '_' + ('beta_kl'+str(temp)) if temp != '' else ''
    readable_name += '_' + ('task_descr' if params_flatten.get('model_params.task_descr_extractor_params', None) else '')
    readable_name += '_' + ('data_encoder' if params_flatten.get('model_params.use_data_encoder', None) else '')
    readable_name += '_' + ('task_var' if params_flatten.get('model_params.use_task_var', None) else '')
    readable_name += '_' + ('use_hloss' if params_flatten.get('model_params.use_hloss', None) else '')

    prefix = readable_name.lower()+uid

    return prefix


def get_dataset_partitions(dataset_name, dataset_params, ds_folder):
    assert dataset_name in loader_dict, 'Unknown dataset'
    return loader_dict[dataset_name](**dataset_params, ds_folder=ds_folder)


def get_model(model_name, model_params):
    assert model_name in metalearnerclass_dict, "unhandled model"
    model_class = metalearnerclass_dict[model_name]

    params = {}
    for k, v in model_params.items():
        if k == 'feature_extractor_params':
            assert isinstance(v, dict) and v['arch'] in inner_class_dict, "unhandled feature extractor"
            fe_class = inner_class_dict[v['arch']]
            fe_params = {i: j for i, j in v.items() if i != 'arch'}
            feature_extractor = fe_class(**fe_params)
            params['feature_extractor'] = feature_extractor
        elif k == 'task_descr_extractor_params':
            if v is None:
                task_descr_extractor = None
            else:
                assert isinstance(v, dict) and v['arch'] in inner_class_dict, "unhandled feature extractor"
                tde_class = inner_class_dict[v['arch']]
                tde_params = {i: j for i, j in v.items() if i != 'arch'}
                task_descr_extractor = tde_class(**tde_params)
            params['task_descr_extractor'] = task_descr_extractor
        else:
            params[k] = v

    return model_class(**params)


def get_config_params(dataset):
    if dataset == 'mhc':
        import few_shot_regression.configs.config_mhc as cfg
    elif dataset == 'bindingdb':
        import few_shot_regression.configs.config_bdb as cfg
    elif dataset == 'movielens':
        import few_shot_regression.configs.config_movielens as cfg
    elif dataset == 'uci':
        import few_shot_regression.configs.config_uci as cfg
    elif dataset == 'toy' or dataset == 'easytoy':
        import few_shot_regression.configs.config_toy as cfg
    else:
        raise Exception("Dataset {} is not found".format(dataset))

    algo_params = dict(mann=cfg.grid_mann,
                       maml=cfg.grid_maml,
                       snail=cfg.grid_snail,
                       deep_prior=cfg.grid_deep_prior,
                       multitask=cfg.grid_multitask,
                       metakrr_sk=cfg.grid_metakrr_sk,
                       metakrr_mk=cfg.grid_metakrr_mk,
                       metagp_sk=cfg.grid_metagp_sk,
                       metagp_mk=cfg.grid_metagp_mk)
    return algo_params


def save_stats(scores_dict, outfile=sys.stdout):
    metrics = list(scores_dict.keys())
    metrics.remove('size')
    sizes = scores_dict['size']

    results = [
        OrderedDict(
            ([('name', dataset_name), ('size', sizes[dataset_name])] +
             [(metric_name + aggregator.__name__, aggregator(scores_dict[metric_name][dataset_name]))
              for metric_name in metrics for aggregator in [np.mean, np.median, np.std]]
             )
        ) for dataset_name in sizes]

    results = pd.DataFrame(results)
    results.to_csv(outfile, index=False, sep='\t')
    return results


if __name__ == '__main__':
    pass