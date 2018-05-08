import torch, os, sys, argparse
import inspect
import pandas as pd
import numpy as np
from utils_data.loaders import load_fewshot_bindingdb, load_fewshot_movielens, load_fewshot_mhcII_DRB
from models import KrrMetaLearner, SNAIL, MAML, MANN, LstmFeaturesExtractor, LstmBasedRegressor, \
    PretrainBase, Cnn1dBasedRegressor, Cnn1dFeaturesExtractor, FcFeaturesExtractor, FcBasedRegressor
from sklearn.model_selection import ParameterGrid
from collections import OrderedDict

SAVING_DIR_FORMAT = '{expts_dir}/results_{dataset_name}_{algo}_{arch}'


def dict2str(d):
    e = [(k, d[k]) for k in sorted(d.keys()) if type(d[k]) != bool]
    base = '_'.join([k for k in sorted(d.keys()) if d[k] is True])
    r = str(e).replace(' ', '').replace(':', '_').replace('\'', '').replace(',', '_').replace('(', '').replace(')', '')
    r = base + '_' + r
    return r


def load_data(dataset_name, k_per_episode, batch_size=10, fold=None):
    if dataset_name == 'bindingdb':
        loader = load_fewshot_bindingdb
    elif dataset_name == 'movielens':
        loader = load_fewshot_movielens
    elif dataset_name == 'mhc':
        loader = load_fewshot_mhcII_DRB
    else:
        raise Exception('Unsupported dataset name')
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
    g = ['krr', 'mann', 'snail']
    if arch == 'cnn':
        inner_class = Cnn1dFeaturesExtractor if algo in g else Cnn1dBasedRegressor
    elif arch == 'lstm':
        inner_class = LstmFeaturesExtractor if algo in g else LstmBasedRegressor
    elif arch == 'fc':
        inner_class = FcFeaturesExtractor if algo in g else FcBasedRegressor
    else:
        raise ValueError("unhandled arch")
    return inner_class


def get_model(algo, arch, params):
    inner_class = get_inner_class(algo, arch)
    if algo == 'krr':
        metalearnerclass = KrrMetaLearner
    elif algo == 'mann':
        metalearnerclass = MANN
    elif algo == 'maml':
        metalearnerclass = MAML
    elif algo == 'snail':
        metalearnerclass = SNAIL
    elif algo == 'pretrain':
        metalearnerclass = PretrainBase
    else:
        raise ValueError("algo's name unhandled")
    temp = set(inspect.signature(inner_class.__init__).parameters.keys())
    inner_params = {k: v for k, v in params.items() if k in temp}
    meta_params = {k: v for k, v in params.items() if k not in temp}
    inner_module = inner_class(**inner_params)
    print(meta_params)
    metalearner = metalearnerclass(inner_module, **meta_params)
    return metalearner


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


def comparison_expts(algo, arch, dataset_name, k_per_episode,
                     fit_params, eval_params, expts_dir, max_episodes=25000,
                     nb_examples_per_epoch=20000,
                     batch_size=64, fit=True, fold=None):
    n_epochs = 1
    steps_per_epoch = int(nb_examples_per_epoch/(batch_size*k_per_episode))
    data_iterator = load_data(dataset_name, k_per_episode, batch_size, fold=fold)

    expts_dir = SAVING_DIR_FORMAT.format(expts_dir=expts_dir, algo=algo,
                                         arch=arch, dataset_name=dataset_name)
    if not os.path.exists(expts_dir):
        os.makedirs(expts_dir, exist_ok=True)

    for meta_train, meta_valid, meta_test in data_iterator:
        res = get_outfile_names(expts_dir, k_per_episode, fit_params, fold)
        log_fname, ckp_fname, result_fname = res
        if hasattr(meta_train, 'ALPHABET_SIZE'):
            fit_params.update(dict(vocab_size=meta_train.ALPHABET_SIZE))
        metalearner = get_model(algo, arch, fit_params)
        if fit:
            metalearner.fit(meta_train, meta_valid, steps_per_epoch=steps_per_epoch,
                            n_epochs=n_epochs, max_episodes=max_episodes, batch_size=batch_size,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', default=0, type=int, help='the number of the job run in this execution')
    parser.add_argument('--nb_jobs', default=-1, type=int, help='The numbers of job scheduled. -1 means all the jobs')
    parser.add_argument('--dataset', default='movielens', type=str, help='The name of the dataset for the experiments')
    parser.add_argument('--outdir', default='expt_results', type=str, help='The name of the output directory for the experiments')
    parser.add_argument('--algos',
                        default=['krr', 'snail', 'mann', 'maml', 'pretrain'],
                        type=str, nargs='+',
                        help='The name of the algos: krr|snail|mann|maml|pretrain')
    args = parser.parse_args()

    algos, part, nb_jobs, dataset = args.algos, args.part, args.nb_jobs, args.dataset
    if dataset == 'mhc':
        from config_mhc import *
    elif dataset == 'bindingdb':
        from config_bdb import *
    elif dataset == 'movielens':
        from config_movielens import *

    algo_dict = {'krr': grid_krr, 'mann': grid_mann, 'maml': grid_maml,
                 'snail': grid_snail, 'pretrain': grid_pretrain}
    algo_grids = [algo_dict[a] for a in algos]

    magic_number = 42
    np.random.seed(magic_number)
    torch.manual_seed(magic_number)
    params_list = list(ParameterGrid(algo_grids))
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

    expts_directory = os.path.join(args.outdir, dataset)
    for param in params_for_this_part:
        print(param)
        comparison_expts(**param, expts_dir=expts_directory, nb_examples_per_epoch=nb_examples_per_epoch,
                         batch_size=batch_size, max_episodes=max_episodes, fit=True)
