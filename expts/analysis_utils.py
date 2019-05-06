import os
import io
import ast
import sys
import glob
import json
import pickle
import itertools
import ipywidgets
import functools
import operator
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from functools import partial
from collections import defaultdict, OrderedDict
from IPython.display import display
from metalearn.datasets.loaders import load_dataset
from metalearn.models.factory import ModelFactory
from metalearn.metric import mse, vse, r2_score, pearsonr
from ivbase.utils.memoize import memoize, hash_dict
mpl.rcParams['font.family'] = 'Arial'
sns.set(rc={'figure.figsize': (10, 6)})
sns.set_style('ticks')
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2})


def unflatten(dictionary):
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict


def save_stats(scores_dict, outfile=sys.stdout):
    metrics = list(scores_dict.keys())
    metrics.remove('size')
    metrics.remove('name')
    names = scores_dict['name']
    sizes = scores_dict['size']

    results = [
        OrderedDict(
            ([('name', names[idx]), ('size', sizes[idx])] +
             [(metric_name + aggregator.__name__, aggregator(scores_dict[metric_name][idx]))
              for metric_name in metrics for aggregator in [np.mean, np.median, np.std]]
             ))
        for idx in names
    ]

    results = pd.DataFrame(results)
    results.to_csv(outfile, index=False, sep='\t')
    return results


def load_model_and_metatest(folder, return_params=False):
    param_file = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('_params.json')]
    model_file = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('_ckp.ckp')]
    if len(param_file) == 0 or len(model_file) == 0:
        return
    param_file = param_file[0]
    model_file = model_file[0]

    with open(param_file) as fd:
        params = json.load(fd)

    params = unflatten(params)
    model_name, model_params = params['model_name'], params['model_params']
    dataset_name, dataset_params = params['dataset_name'], params['dataset_params']
    model = ModelFactory()(model_name, **model_params)
    model.load(model_file)
    _, _, meta_test = load_dataset(dataset_name, **dataset_params)
    if return_params:
        return model, meta_test, params
    return model, meta_test


def all_files_ending_with(folder, extension):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(extension)]


def update_params_for_mk(params):
    keyr = 'input_features_extractor_params'
    new_dict = dict()
    keys = []
    for k, v in params.items():
        if keyr in k:
            new_dict[k.replace(keyr, 'feature_extractor_params')] = v
            keys.append(k)
    for k in keys:
        params.pop(k)
    params.update(new_dict)

    if 'model_params.condition_on' in params:
        return params
    key1, key2, new_key = ('model_params.dataset_encoder_params.', 'model_params.task_descr_extractor_params.', 'model_params.condition_on')
    keys = params.keys()
    key1_exists = any(map((lambda x: key1 in x), keys))
    key2_exists = any(map((lambda x: key2 in x), keys))
    if key1_exists and key2_exists:
        params[new_key] = 'both'
    elif key1_exists:
        params[new_key] = 'train_data'
    elif key2_exists:
        params[new_key] = 'task_descr'
    else:
        params[new_key] = '_none'

    return params


def get_result_summaries(results_folder, return_test_filenames=True):
    logs_ext = '_log.log'
    params_ext = '_params.json'
    res_ext = '_res.csv'
    res_files = all_files_ending_with(results_folder, res_ext)
    data_list = []
    for res_file in res_files:
        param_file = res_file.replace(res_ext, params_ext)
        log_file = res_file.replace(res_ext, logs_ext)
        with open(param_file) as fd:
            params = json.load(fd)
            params = {k: str(v) for k, v in params.items()}
            params = update_params_for_mk(params)
        vloss = pd.read_csv(log_file, sep='\t').val_loss.min()
        params.update(dict(val_loss=vloss))
        res_data = pd.read_csv(res_file, sep='\t')
        tboard_folder = res_file.replace(res_ext, '')
        events_file = [os.path.join(tboard_folder, f) for f in os.listdir(tboard_folder) if f.startswith('events')]
        if len(events_file) != 0:
            events_file = events_file[0]
        else:
            events_file = None
        try:
            params.update(dict(
                pcc=res_data.pccmean.mean(),
                r2=res_data.r2mean.mean(),
                mse=res_data.msemean.mean(),
                tboard_folder=tboard_folder,
                events_file=events_file
            ))
        except:
            params.update(dict(
                pcc=res_data.pearsonrmean.mean(),
                r2=res_data.r2_scoremean.mean(),
                mse=res_data.msemean.mean(),
                tboard_folder=tboard_folder,
                events_file=events_file
            ))
        data_list.append(params)
    test_filenames = sorted(res_data.name.str.split('/').apply(lambda x: x[-1]))
    data = pd.DataFrame(data_list)
    data = data[data['dataset_params.max_tasks'] == 'None']
    data = data.loc[:, data.apply(pd.Series.nunique) != 1]
    data = data.fillna('None')
    data.rename(lambda x: str(x).split('.')[-1], axis='columns', inplace=True)
    if return_test_filenames:
        return data, test_filenames
    return data


def dropdown_widget(name, opts):
    return ipywidgets.Dropdown(
        options=opts,
        value=opts[0],
        description=name,
        disabled=False)


def plot_heatmap(resdata, metric, column_names, row_names, filters=None, save_folder=None):
    table = resdata.copy()
    if filters is not None:
        for key, value in filters.items():
            table = table[table[key] == value]

    table = pd.pivot_table(table, values=metric, index=row_names, columns=column_names)
    out = ipywidgets.Output()
    with out:
        fig, ax = plt.subplots(figsize=(table.shape[1], table.shape[0]))
        ax = sns.heatmap(table, annot=True, fmt=".3f", cmap="Blues", square=True, ax=ax)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=80)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=60)
        plt.title("Heatmap of {}".format(metric))
        if save_folder is not None:
            filter_str = '--'.join([str(k) + '-' + str(v) for k, v in filters.items()])
            plt.savefig(os.path.join(res_folder, f'heatmaps_{metric}_{filter_str}.svg'), dpi=800)
        return out


def get_metric_heatmap_widgets(data, filter_names):
    widgets = dict(metric=dropdown_widget(name='Metric', opts=['mse', 'r2', 'pcc', 'val_loss']))
    if filter_names is not None:
        print(filter_names)
        for key in filter_names:
            widgets[key] = dropdown_widget(name=key.replace('_', ' ').capitalize(), opts=list(set(data[key].tolist())))
    return widgets


def get_heatmaps(data, heat_column_names, heat_row_names, metric, **kwargs):
    output = plot_heatmap(data,
                          metric=metric,
                          column_names=heat_column_names,
                          row_names=heat_row_names,
                          filters=kwargs)
    return output


def plot_fitting_curves(resdata, **filters):
    table = resdata.copy()
    test_filename = filters.pop('test_filename', None)
    config = ast.literal_eval(filters.pop('configs'))
    for key, value in config.items():
        table = table[table[key] == value]

    events_file = table['events_file'].tolist()[0]
    test_name = 'test' + test_filename.split('_')[-1].split('.')[0]
    out = ipywidgets.Output()
    with out:
        print(test_name, table['events_file'].tolist())
        for evt in tf.train.summary_iterator(events_file):
            for value in evt.summary.value:
                if test_name == value.tag:
                    imageBinaryBytes = (value.image.encoded_image_string)
                    stream = io.BytesIO(imageBinaryBytes)
                    img = Image.open(stream)
                    fig = plt.figure(dpi=120)
                    plt.axis('off')
                    plt.imshow(img)
                    plt.show()
    return out


def get_fitting_curves_widgets(data, test_filenames):
    metrics = ['mse', 'r2', 'pcc', 'val_loss', 'tboard_folder', 'tboard_folder', 'events_file']
    filter_names = list(set(data.columns.tolist()).difference(metrics))
    config_options = [str(row.to_dict()) for _, row in data[filter_names].iterrows()]

    widgets = dict()
#     for key in filter_names:
#         widgets[key] = dropdown_widget(name=key.replace('_', ' ').capitalize(), opts=list(set(data[key].tolist())))
    widgets['configs'] = dropdown_widget(name='Configuration', opts=config_options)
    widgets['test_filename'] = dropdown_widget(name='Test filename', opts=test_filenames)
    return widgets
