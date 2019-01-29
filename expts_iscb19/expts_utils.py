import sys, hashlib, json, os
import pandas as pd
import numpy as np
from collections import OrderedDict, MutableMapping
from metalearn.datasets.loaders import load_dataset
from metalearn.models.factory import ModelFactory
from metalearn.utils.metric import mse, vse, r2, pcc


SAVING_DIR_FORMAT = '{expts_dir}/results_{dataset_name}_{algo}_{arch}'


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
    return uid


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


def run_experiment(model_name, model_params, dataset_name, dataset_params,
                   fit_params, output_path, input_path=None):
    all_params = locals()
    data_path = all_params['input_path']
    del all_params['output_path'], all_params['input_path']

    out_prefix = get_outfname_prefix(all_params)
    ckp_fname = "{}/{}_ckp.ckp".format(output_path, out_prefix)
    result_fname = "{}/{}_res.csv".format(output_path, out_prefix)
    log_fname = "{}/{}_log.log".format(output_path, out_prefix)
    tboard = "{}/{}".format(output_path, out_prefix)

    with open("{}/{}_params.json".format(output_path, out_prefix), 'w') as fd:
        temp = flatten_dict(all_params)
        temp = {k: temp[k] for k in temp 
            if isinstance(temp[k], (str, int, float, bool, list, dict, tuple, type(None)))}
        json.dump(temp, fd, indent=4, sort_keys=True)

    model = ModelFactory()(model_name, **model_params)
    meta_train, meta_valid, meta_test = load_dataset(dataset_name, **dataset_params)

    if os.path.exists(ckp_fname):
        try:
            model.load(ckp_fname)
        except:
            pass
    fit_params.update(dict(log_filename=log_fname,
                           checkpoint_filename=ckp_fname,
                           tboard_folder=tboard))
    model.fit(meta_train, meta_valid, **fit_params)

    scores = model.evaluate(meta_test, metrics=[mse, vse, r2, pcc])
    with open(result_fname, "w") as outfile:
        results = save_stats(scores, outfile)
        print(results)