import torch, os, csv, sys
import inspect
import numpy as np
from utils.loaders import load_fewshot_bindingdb, load_fewshot_mhcII, load_fewshot_mhcII_DRB
from models import KrrMetaLearner, MAML, MANN, LstmFeaturesExtractor, LstmBasedRegressor, PretrainBase, Cnn1dBasedRegressor, Cnn1dFeaturesExtractor
from sklearn.model_selection import ParameterGrid

SAVING_DIR_FORMAT = '{expts_dir}/results_{dataset_name}_{algo}_{arch}'


def dict2str(d):
    e = [(k, d[k]) for k in sorted(d.keys()) if type(d[k]) != bool]
    base = '_'.join([k for k in sorted(d.keys()) if d[k] is True])
    r = str(e).replace(' ', '').replace(':', '_').replace('\'', '').replace(',', '_').replace('(', '').replace(')', '')
    r = base + '_' + r
    return r


def load_data(dataset_name, max_examples_per_episode, batch_size=10, fold=None):
    if dataset_name == 'bindingdb':
        loader = load_fewshot_bindingdb
    elif dataset_name == 'mhc':
        loader = load_fewshot_mhcII
    elif dataset_name == 'mhcpan':
        loader = load_fewshot_mhcII_DRB
    else:
        raise Exception('Unsupported dataset name')
    if fold:
        return loader(max_examples_per_episode, batch_size, fold=fold)
    else:
        return loader(max_examples_per_episode, batch_size)


def get_outfile_names(expts_dir, algo, arch, dataset_name, max_examples_per_episode, params, fold=0):
    temp = dict2str(params)
    format_params = (expts_dir, algo, arch, dataset_name, fold, max_examples_per_episode, temp)
    log_fname = "{}/log_{}_{}_{}_fold{}_nbsamples{}--{}.txt".format(*format_params)
    ckp_fname = "{}/ckp_{}_{}_{}_fold{}_nbsamples{}--{}.ckp".format(*format_params)
    result_fname = "{}/results_{}_{}_{}_fold{}_nbsamples{}--{}.txt".format(*format_params)
    return log_fname, ckp_fname, result_fname


def get_model(algo, arch, params):
    if algo in ['krr', 'mann']:
        inner_class = Cnn1dFeaturesExtractor if arch == 'cnn' else LstmFeaturesExtractor
    else:
        inner_class = Cnn1dBasedRegressor if arch == 'cnn' else LstmBasedRegressor
    if algo == 'krr':
        metalearnerclass = KrrMetaLearner
    elif algo == 'mann':
        metalearnerclass = MANN
    elif algo == 'maml':
        metalearnerclass = MAML
    elif algo == 'pretrain':
        metalearnerclass = PretrainBase
    else:
        raise ValueError("algo's name unhandled")

    temp = set(inspect.signature(inner_class.__init__).parameters.keys())
    inner_params = {k: v for k, v in params.items() if k in temp}
    meta_params = {k: v for k, v in params.items() if k not in temp}
    inner_module = inner_class(**inner_params)
    metalearner = metalearnerclass(inner_module, **meta_params)
    return metalearner


def save_stats(scores, outfile=sys.stdout):
    keys = ['name', 'size', 'r2_mean', 'r2_median', 'r2_std', 'pcc_mean', 'pcc_median', 'pcc_std']
    results = dict([(key, []) for key in keys])

    r2, pcc, sizes = scores
    for dataset in sizes:
        results['name'] += [dataset.split('/')[-1].split('.')[0]]
        results['size'] += [sizes[dataset]]
        results['r2_mean'] += [np.mean(r2[dataset])]
        results['r2_median'] += [np.median(r2[dataset])]
        results['r2_std'] += [np.std(r2[dataset])]
        results['pcc_mean'] += [np.mean(pcc[dataset])]
        results['pcc_std'] += [np.std(pcc[dataset])]
        results['pcc_median'] += [np.median(pcc[dataset])]

    outfile.write('\t'.join(keys)+'\n')
    temp = []
    for key in keys:
        if type(results[key][0]) == float:
            temp.append(np.round(results[key]))
        else:
            temp.append(results[key])
    temp = zip(*temp)
    for line in temp:
        outfile.write('\t'.join(map(str, line)) + '\n')
    return results


def comparison_expts(algo, arch, dataset_name, max_examples_per_episode,
                     fit_params, eval_params, expts_dir, fit=True, fold=None):
    batch_size, n_epochs = 1, 1000
    nb_examples_per_epoch = 20000 if 'mhc' in dataset_name else 50000
    steps_per_epoch = 1 # int(nb_examples_per_epoch/(batch_size*max_examples_per_episode))
    data_iterator = load_data(dataset_name, max_examples_per_episode, batch_size, fold=fold)

    expts_dir = SAVING_DIR_FORMAT.format(expts_dir=expts_dir, algo=algo, arch=arch, dataset_name=dataset_name)
    if not os.path.exists(expts_dir):
        os.makedirs(expts_dir, exist_ok=True)

    for meta_train, meta_valid, meta_test in data_iterator:
        res = get_outfile_names(expts_dir, algo, arch, dataset_name, max_examples_per_episode, fit_params, fold)
        log_fname, ckp_fname, result_fname = res
        fit_params.update(dict(vocab_size=meta_train.ALPHABET_SIZE))
        metalearner = get_model(algo, arch, fit_params)
        if fit:
            metalearner.fit(meta_train, meta_valid, steps_per_epoch=steps_per_epoch, n_epochs=n_epochs,
                            log_filename=log_fname, checkpoint_filename=ckp_fname)
        else:
            metalearner.load(ckp_fname)

        if eval_params is None:
            scores = metalearner.evaluate(meta_test)
            with open(result_fname, "w") as outfile:
                results = save_stats(scores, outfile)
                print(results)
        else:
            with open(result_fname, "w") as outfile:
                for p in ParameterGrid(eval_params):
                    print(p)
                    scores = metalearner.evaluate(meta_test, **p)
                    outfile.write(dict2str(p)+'\n')
                    results = save_stats(scores, outfile)
                    print(results)


if __name__ == '__main__':
    from config_mhc import *
    if len(sys.argv[1:]) > 0:
        part = int(sys.argv[1])
        nb_jobs = int(sys.argv[2])
    else:
        part = 0
        nb_jobs = 1

    magic_number = 42
    np.random.seed(magic_number)
    torch.manual_seed(magic_number)
    params_list = list(ParameterGrid([grid_krr_cnn]))
    params_list = list(ParameterGrid([grid_mann_cnn, grid_maml_cnn,
                                      grid_krr_cnn, grid_pretrain_cnn]))
    np.random.shuffle(params_list)
    nb_jobs = len(params_list) if nb_jobs == -1 else nb_jobs
    if len(params_list) % nb_jobs == 0:
        nb_per_part = int(len(params_list)/nb_jobs)
    else:
        nb_per_part = int(len(params_list)/nb_jobs) + 1
    start_index, end_index = part*nb_per_part, (part+1)*nb_per_part
    if start_index > len(params_list):
        exit()
    print(start_index, end_index)
    params_for_this_part = params_list[start_index: end_index]
    for param in params_for_this_part:
        print(param)
        comparison_expts(**param, expts_dir=expts_directory, fit=True)
