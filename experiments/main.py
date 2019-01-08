import torch, os, argparse
from sklearn.model_selection import ParameterGrid
from expts_utils import *
from metalearn.utils.metric import mse, vse, r2, pcc
from metalearn.datasets.loaders import load_dataset
from metalearn.models.factory import ModelFactory
from configs.factory import ConfigFactory



def run_experiment(model_name, model_params, dataset_name, dataset_params,
                   fit_params, expts_dir, refit=True):
    all_params = locals()
    del all_params['expts_dir'], all_params['refit']
    model = ModelFactory()(model_name, **model_params)
    meta_train, meta_valid, meta_test = load_dataset(dataset_name, **dataset_params)

    out_prefix = os.path.join(expts_dir, get_outfname_prefix(all_params))
    ckp_fname, result_fname = "{}_ckp.ckp".format(out_prefix), "{}_res.csv".format(out_prefix),
    save_params(all_params, "{}_param.json".format(out_prefix))
    if os.path.exists(ckp_fname):
        try:
            model.load(ckp_fname)
        except:
            pass
    fit_params.update(dict(log_filename="{}_log.log".format(out_prefix),
                           checkpoint_filename=ckp_fname,
                           tboard_folder=out_prefix))
    fit_and_eval(model, meta_train, meta_valid, meta_test, fit_params, result_fname, refit=refit)


def fit_and_eval(model, meta_train, meta_valid, meta_test, fit_params, result_fname, refit=False):
    refit = False
    if refit and model.is_fitted:
        pass
    else:
        model.fit(meta_train, meta_valid, **fit_params)

    scores = model.evaluate(meta_test, metrics=[mse, vse, r2, pcc])
    with open(result_fname, "w") as outfile:
        results = save_stats(scores, outfile)
        print(results)


if __name__ == '__main__':
    from pprint import pprint
    magic_number = 42
    np.random.seed(magic_number)
    torch.manual_seed(magic_number)

    parser = argparse.ArgumentParser()
    parser.add_argument('--part', default=0, type=int, help='the number of the job run in this execution')
    parser.add_argument('--dataset', default='easytoy', type=str, help='The name of the dataset for the experiments')
    parser.add_argument('--outdir', default='results/test', type=str, help='The name of the output directory for the experiments')
    parser.add_argument('--algos',
                        default=['mars'],
                        type=str, nargs='+',
                        help='The name of the algos tested')
    args = parser.parse_args()
    algos, part, dataset, expts_directory = args.algos, args.part, args.dataset, args.outdir

    algo_dict = ConfigFactory()(dataset)
    algo_grids = [algo_dict[a] for a in algos]
    params_list = list(ParameterGrid(algo_grids))
    np.random.shuffle(params_list)

    param_for_this_part = params_list[part]
    param_for_this_part['dataset_name'] = dataset
    pprint(param_for_this_part)
    if not os.path.exists(expts_directory):
        os.makedirs(expts_directory, exist_ok=True)
    run_experiment(**param_for_this_part, expts_dir=expts_directory, refit=True)