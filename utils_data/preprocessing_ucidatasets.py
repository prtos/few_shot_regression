import os
import pandas as pd
import numpy as np
from collections import OrderedDict
from scipy.io.arff import loadarff
from scipy.stats import chi2_contingency, kurtosis, skew, gmean
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import GridSearchCV, KFold, St
from sklearn.svm import SVC
from quickviz import heatmap


UCI_FOLDER = "../datasets/uci"
UCI_CROSSVAL_FOLDER = UCI_FOLDER+'_rbf'
if not os.path.exists(UCI_CROSSVAL_FOLDER):
    os.makedirs(UCI_CROSSVAL_FOLDER, exist_ok=True)


def entropy(arr, arr2=None):
    if arr2 is None:
        c = np.histogram(arr)[0]
    else:
        c = np.histogram2d(arr, arr2)[0]
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -np.sum(c_normalized * np.log2(c_normalized))
    return H


def mutual_information(x, y):
    return entropy(x) + entropy(y) - entropy(x, y)


def get_meta_features(inputs, targets):
    # Cross-Disciplinary Perspectives on Meta-Learning for Algorithm Selection KATE A. SMITH-MILES
    nb_binary_attr = np.sum([len(np.unique(col)) == 2 for col in inputs.T])
    corrs = np.triu(np.corrcoef(inputs, rowvar=False)).flatten()
    total_variation = np.std(inputs, axis=0) / (np.mean(inputs, axis=0) + 1e-4)
    class_entropy = entropy(targets)
    inputs_entropy = np.mean(np.apply_along_axis(entropy, axis=0, arr=inputs))
    mi_class_attr = np.mean(np.apply_along_axis(lambda x: entropy(targets, x), axis=0, arr=inputs))
    CCA(n_components=1).fit(inputs, targets)
    metadata = OrderedDict(
        # simples
        nsamples=inputs.shape[0],
        nfeatures=inputs.shape[1],
        nclasses=len(np.unique(targets)),
        n_bin_attr=nb_binary_attr,

        # statistical
        geom_mean_std_ratio=gmean(total_variation),
        attr_mean_corr=np.mean(np.abs(corrs)),
        # first_canonical_corr=0,
        # first_fract_sep=0,
        attr_skewness=np.mean(skew(inputs, axis=0)),
        attr_kurtosis=np.mean(kurtosis(inputs, axis=0)),

        # information theory
        class_entropy=class_entropy,
        attr_mean_entropy=inputs_entropy,
        mean_mutual_info=mi_class_attr,
        equiv_number_attr=class_entropy/mi_class_attr,
        noise_signal_ratio=(inputs_entropy - mi_class_attr)/mi_class_attr
    )
    return metadata


def crossvalidate(inputs, targets):
    # SVM with rbf kernel
    n = 15
    param_grid = dict(kernel=['rbf'],
                      gamma=2**np.linspace(-n, n, 2*n+1),
                      C=2**np.linspace(-n, n, 2*n+1))

    cv = KFold(n_splits=10, random_state=42)
    grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv, n_jobs=-1)
    grid.fit(inputs, targets)
    cv_results = grid.cv_results_
    selected_columns = [el for el in cv_results.keys()
                        if el.startswith('param_') or (el in ['mean_test_score', 'std_test_score'])]
    cv_results = pd.DataFrame(cv_results)[selected_columns]
    print(cv_results.head(5))
    print(grid.best_params_, grid.best_score_)
    return cv_results


def caracterize_learn_and_save_results(inputs, targets, output_prefix):
    metadata = get_meta_features(inputs, targets)
    print(metadata)
    results = crossvalidate(inputs, targets)
    results.to_csv(output_prefix + '.csv', index=False)
    heatmap(results, prefix_filename=output_prefix)


def create_few_shot_datasets(max_datasize=5e4):
    filenames = [f for f in os.listdir(UCI_FOLDER) if f.endswith('arff')]
    for arff_filename in filenames:
        output_prefix = os.path.join(UCI_CROSSVAL_FOLDER, arff_filename[:-5])
        inputs, targets = load_arff_dataset(os.path.join(UCI_FOLDER, arff_filename))
        m = len(targets)
        if m < max_datasize:
            caracterize_learn_and_save_results(inputs, targets, output_prefix)
        else:
            n_splits = m / max_datasize
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for i, (_, train_index) in enumerate(splitter.split(inputs)):
                inputs_split, targets_split = inputs[train_index], targets[train_index]
                caracterize_learn_and_save_results(inputs_split, targets_split, output_prefix+'_split{}'.format(i))


def load_arff_dataset(arff_filename):
    numeric_type, nominal_type = 'numeric', 'nominal'
    print(arff_filename)
    data, meta = loadarff(arff_filename)
    N = data.shape[0]
    inputs, targets = data[meta.names()[:-1]], data[meta.names()[-1]]
    non_numeric_attrs = [name for name, typ in zip(meta.names()[:-1], meta.types()[:-1])
                         if typ != numeric_type]
    numeric_attrs = [name for name, typ in zip(meta.names()[:-1], meta.types()[:-1])
                     if typ == numeric_type]
    targets_type = meta.types()[-1]
    if targets_type != numeric_type:
        targets = LabelEncoder().fit_transform(targets)

    other_inputs = inputs[numeric_attrs]
    other_inputs = np.array([list(datapoint) for datapoint in other_inputs])

    if non_numeric_attrs != []:
        categoric_inputs = inputs[non_numeric_attrs]
        categoric_inputs = np.array([LabelEncoder().fit_transform(categoric_inputs[name])
                                     for name in categoric_inputs.dtype.names]).T
        mask_binary = np.array([len(np.unique(col)) != 2 for col in categoric_inputs.T])
        categoric_inputs = OneHotEncoder(categorical_features=mask_binary, sparse=False).fit_transform(categoric_inputs)
        if numeric_attrs != []:
            inputs = np.concatenate((other_inputs, categoric_inputs), axis=1)
        else:
            inputs = categoric_inputs
    else:
        inputs = other_inputs
    if inputs.shape[0] != N:
        raise Exception('number of training examples incorrect')
    non_na_rows = np.logical_not(np.any(np.isnan(inputs), axis=1)).nonzero()[0]
    inputs, targets = inputs[non_na_rows], targets[non_na_rows]
    inputs = inputs[:, inputs.std(axis=0).nonzero()[0]]
    return inputs, targets


if __name__ == '__main__':
    create_few_shot_datasets()