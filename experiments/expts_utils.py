import sys, hashlib, json
import pandas as pd
import numpy as np
from collections import OrderedDict, MutableMapping


SAVING_DIR_FORMAT = '{expts_dir}/results_{dataset_name}_{algo}_{arch}'


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


def save_params(params, fname):
    with open(fname, 'w') as fd:
        json.dump(params, fd, indent=2)


def load_nested_params(fd):
    if hasattr(fd, 'read'):
        content = fd.read()
        res = ast.literal_eval(content)
        # res = json.load(fd)
    elif isinstance(fd, str) and (
                    (fd.startswith('{') and fd.endswith('}')) or
                    (fd.startswith('[') and fd.endswith(']'))):
        res = ast.literal_eval(fd)
    else:
        res = fd
    if isinstance(res, dict):
        res = {k: (load_nested_params(v)
                   if isinstance(v, str) and ((v.startswith('{') and v.endswith('}')) or
                        (v.startswith('[') and v.endswith(']')))
                   else v)
               for k, v in res.items()}
    return res


if __name__ == '__main__':
    import ast
    load_nested_params("""{"krr": "[{'kernel': ['rbf'], 'alpha': array([1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02]), 'gamma': array([1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02])}, {'kernel': ['linear'], 'alpha': array([1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02])}]", "svm": "{'C': [1.0, 10.0, 100.0, 1000.0], 'gamma': array([1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02])}", "rf": "{'n_estimators': [100, 200]}", "target": "Inhibition", "aid": "1241454.csv"}""")
    hps = {'dataset_name': 'uci',
           'dataset_params': {'max_examples_per_episode': 10, 'batch_size': 32},
           'fit_params': {'valid_size': 0.25, 'n_epochs': 100, 'steps_per_epoch': 500},
           'model_name': 'metagp_sk',
           'model_params': {'feature_extractor_params': {'arch': 'fc', 'hidden_sizes': [64, 64],
                                                         'input_size': 2, 'normalize_features': False},
                            'l2': 0.1, 'lr': 0.001}}

    hyperparameters = {str(k): str(v) for (k, v) in hps.items()}

    print(ast.literal_eval(str(hyperparameters)))
    print(load_nested_params(json.dumps(hyperparameters)))
    pass