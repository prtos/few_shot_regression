import sys
import inspect
import pandas as pd
import numpy as np
from few_shot_regression.utils_data.loaders import load_fewshot_bindingdb, load_fewshot_movielens, load_fewshot_mhcII_DRB
from few_shot_regression.models import KrrMetaLearner, SNAIL, MAML, MANN, LstmFeaturesExtractor, LstmBasedRegressor, MultiTaskLearner,\
    Cnn1dBasedRegressor, Cnn1dFeaturesExtractor, FcFeaturesExtractor, FcBasedRegressor
from collections import OrderedDict

SAVING_DIR_FORMAT = '{expts_dir}/results_{dataset_name}_{algo}_{arch}'
loader_dict = dict(
    bindingdb=load_fewshot_bindingdb,
    movielens=load_fewshot_movielens,
    mhc=load_fewshot_mhcII_DRB,
    uci=load_fewshot_mhcII_DRB)
inner_class_dict = dict(
    cnn=dict(extractor=Cnn1dFeaturesExtractor, regressor=Cnn1dBasedRegressor),
    lstm=dict(extractor=LstmFeaturesExtractor, regressor=LstmBasedRegressor),
    fc=dict(extractor=FcFeaturesExtractor, regressor=FcBasedRegressor))
metalearnerclass_dict = dict(
    mann=MANN, maml=MAML, snail=SNAIL,
    multitask=MultiTaskLearner, metakrr=KrrMetaLearner, fskrr=KrrMetaLearner,
)


def dict2str(d):
    e = [(k, d[k]) for k in sorted(d.keys()) if type(d[k]) != bool]
    base = '_'.join([k for k in sorted(d.keys()) if d[k] is True])
    r = str(e).replace(' ', '').replace(':', '_').replace('\'', '').replace(',', '_').replace('(', '').replace(')', '')
    r = base + '_' + r
    return r


def load_data(dataset_name, k_per_episode, batch_size=10, fold=None):
    if dataset_name not in loader_dict:
        raise Exception('Unknown dataset ')
    loader = loader_dict[dataset_name]
    if fold:
        return loader(k_per_episode, batch_size, fold=fold)
    else:
        return loader(k_per_episode, batch_size)


def get_outfile_names(expts_dir, k_per_episode, params, fold=0):
    temp = dict2str(params)
    format_params = (expts_dir, fold, k_per_episode, temp)
    log_fname = "{}/log_fold{}_kshot{}--{}.txt".format(*format_params)
    ckp_fname = "{}/ckp_fold{}_kshot{}--{}.ckp".format(*format_params)
    result_fname = "{}/results_fold{}_kshot{}--{}.txt".format(*format_params)
    return log_fname, ckp_fname, result_fname


def get_inner_class(algo, arch):
    g = ['fskrr', 'metakrr', 'mann', 'snail', 'multitask']
    if arch not in inner_class_dict:
        raise ValueError("unhandled arch")
    inner_class = inner_class_dict[arch]['extractor'] if algo in g else inner_class_dict[arch]['regressor']
    return inner_class


def get_model(algo, arch, params):
    inner_class = get_inner_class(algo, arch)
    if algo not in metalearnerclass_dict:
        raise ValueError("algo's name unhandled")
    temp = set(inspect.signature(inner_class.__init__).parameters.keys())
    inner_params = {k: v for k, v in params.items() if k in temp}
    meta_params = {k: v for k, v in params.items() if k not in temp}
    inner_module = inner_class(**inner_params)
    metalearner = metalearnerclass_dict[algo](inner_module, **meta_params)
    return metalearner


def get_config_params(dataset):
    if dataset == 'mhc':
        import few_shot_regression.configs.config_mhc as cfg
        # import grid_fskrr, grid_metakrr, grid_mann, grid_maml, grid_snail, grid_multitask
    elif dataset == 'bindingdb':
        import few_shot_regression.configs.config_bdb as cfg
    elif dataset == 'movielens':
        import few_shot_regression.configs.config_movielens as cfg
    else:
        raise Exception("Dataset {} is not found".format(dataset))

    algo_dict = {'fskrr': cfg.grid_fskrr, 'metakrr': cfg.grid_metakrr, 'mann': cfg.grid_mann,
                 'maml': cfg.grid_maml, 'snail': cfg.grid_snail, 'multitask': cfg.grid_multitask}

    other_params = dict(nb_examples_per_epoch=cfg.nb_examples_per_epoch,
                        batch_size=cfg.batch_size, max_episodes=cfg.max_episodes)
    return algo_dict, other_params


def save_stats(scores, outfile=sys.stdout):
    mse, r2, pcc, sizes = scores
    results = [
        OrderedDict([('name', dataset.split('/')[-1].split('.')[0]), ('size', sizes[dataset]),
                     ('r2_mean', np.mean(r2[dataset])), ('r2_median', np.median(r2[dataset])),
                     ('r2_std', np.std(r2[dataset])), ('pcc_mean', np.mean(pcc[dataset])),
                     ('pcc_std', np.std(pcc[dataset])), ('pcc_median', np.median(pcc[dataset])),
                     ('mse_mean', np.mean(mse[dataset])), ('mse_std', np.std(mse[dataset])),
                     ('mse_median', np.median(mse[dataset]))])
        for dataset in sizes]
    results = pd.DataFrame(results)
    results.to_csv(outfile, index=False, sep='\t')
    return results

if __name__ == '__main__':
    pass