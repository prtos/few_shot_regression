import os
import pandas as pd
import numpy as np
from collections import OrderedDict, Counter
from scipy.io.arff import loadarff
from scipy.stats import chi2_contingency, kurtosis, skew, gmean
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
import sklearn.metrics as metrics
import sklearn.preprocessing as prep
from quickviz import heatmap


UCI_FOLDER = "../../datasets/originals/uci"
UCI_CROSSVAL_FOLDER = os.path.join(os.path.dirname(os.path.dirname(UCI_FOLDER)), 'uci_rbf')
if not os.path.exists(UCI_CROSSVAL_FOLDER):
    os.makedirs(UCI_CROSSVAL_FOLDER, exist_ok=True)
NB_FOLD = 10
MAX_DATASIZE = int(5e3)


def normalize_task_descr():
    root = UCI_CROSSVAL_FOLDER
    temp = [os.path.join(root, fname) for fname in os.listdir(root)]
    content = np.array([pd.read_csv(fname, nrows=1).values.flatten() for fname in temp])
    content[np.isnan(content)] = 0
    std = content.std(0)
    std[std == 0] = 1
    content = (content - content.mean(0))/std
    for pos, fname in enumerate(temp):
        top = pd.read_csv(fname, nrows=1)
        bottom = pd.read_csv(fname, skiprows=2)
        new_top = pd.DataFrame(dict(zip(top.columns, [[el] for el in content[pos]])))
        new_top.to_csv(fname, index=False)
        bottom.to_csv(fname, index=False, mode='a')


def weighted_f1(y_true, y_pred):
    inter = list(set(y_true).intersection(set(y_pred)))
    if len(inter) <= 1:
        return 0
    return metrics.f1_score(y_true, y_pred, average='weighted', labels=inter)


def weighted_acc(y_true, y_pred):
    raise NotImplementedError()


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
    total_variation = np.abs(np.std(inputs, axis=0) / (np.mean(inputs, axis=0) + 1e-4))
    class_entropy = entropy(targets)
    inputs_entropy = np.mean(np.apply_along_axis(entropy, axis=0, arr=inputs))
    mi_class_attr = np.mean(np.apply_along_axis(lambda x: entropy(targets, x), axis=0, arr=inputs))
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

    cv = KFold(n_splits=5, random_state=42)
    grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv,
                        scoring=dict(
                            acc=metrics.make_scorer(metrics.accuracy_score),
                            f1=metrics.make_scorer(weighted_f1)),
                        refit='acc',
                        n_jobs=-1)
    grid.fit(inputs, targets)
    cv_results = grid.cv_results_
    selected_columns = [el for el in cv_results.keys()
                        if el.startswith(('param_', 'mean_test', 'std_test_'))]

    cv_results = pd.DataFrame({k: cv_results[k] for k in cv_results.keys() if k in selected_columns})
    print(grid.best_params_, grid.best_score_)
    # # print(cv_results.columns)
    return cv_results


def caracterize_learn_and_save_results(inputs, targets, output_prefix):
    output_filename = output_prefix + '.csv'
    if os.path.exists(output_filename):
        return
    metadata = get_meta_features(inputs, targets)
    metadata = pd.DataFrame([metadata])
    metadata.to_csv(output_filename, index=False)
    print(metadata)
    results = crossvalidate(inputs, targets)
    results.to_csv(output_filename, index=False, mode='a')
    heatmap(results, prefix_filename=output_prefix)


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


def random_classes_selection_and_weighting(targets, min_sample_per_class, rgn=None):
    """
    Given the target vector, this function outputs a dict which keys are the classes in
    the vector and the values are their weigths.

    A weight of 0 means that the class is not selected
    :param targets: the target vector
    :param min_sample_per_class: Minimum number of sample that a class need to have to be selected
    :param rgn: a random number generator
    :return: (dict) the weights of every class in the target vector
    """
    rgn = np.random.RandomState() if rgn is None else rgn
    classes = np.unique(targets)
    effective_classes = np.array([k for k, v in Counter(targets).items() if v > min_sample_per_class])
    nb_effective_classes = len(effective_classes)
    if nb_effective_classes < 2:
        return None
    class_weights = {c: (1 if c in effective_classes else 0) for c in classes}
    class_weights = {c: class_weights[c]*(np.exp(rgn.uniform())) for c in class_weights}
    chosen_classes = rgn.choice(effective_classes, size=rgn.choice(np.arange(2, nb_effective_classes+1)), replace=False)
    class_weights = {c: (class_weights[c] if c in chosen_classes else 0) for c in class_weights}
    is_balanced = rgn.binomial(1, 0.55) # 0.45 is the probability of success (success == 1 and failure == 0)
    if is_balanced:
        class_weights = {c: (1 if class_weights[c] > 0 else 0) for c in class_weights}
    s = np.sum(list(class_weights.values()))
    class_weights = {c: class_weights[c] / s for c in class_weights}
    assert np.sum(np.array(list(class_weights.values()))>0) >=2, str(class_weights)+str(effective_classes)
    return class_weights


def random_samples_selection(targets, class_weights, approx_min_sample_per_class, rgn=None, max_datasize=MAX_DATASIZE):
    rgn = np.random.RandomState() if rgn is None else rgn
    samples_weights = np.array([class_weights[el] for el in targets])
    samples_weights /= np.sum(samples_weights)
    nb_classes = sum([v > 0 for v in class_weights.values()])
    nmin = nb_classes * approx_min_sample_per_class
    nmax = min(np.sum(samples_weights > 0), max_datasize)
    if nmin >= nmax:
        return None
    nsamples_sampled = rgn.choice(np.arange(nmin, nmax))
    idx_sampled = rgn.choice(np.arange(len(targets)), size=nsamples_sampled, replace=False, p=samples_weights)
    return idx_sampled


def random_features_selection(inputs, rgn=None):
    rgn = np.random.RandomState() if rgn is None else rgn
    nfeatures = inputs.shape[1]
    idx_features = np.arange(nfeatures)
    rgn.shuffle(idx_features)
    nfeatures_sampled = max(2, rgn.choice(nfeatures))
    return idx_features[:nfeatures_sampled]


def random_features_transformation(inputs, rgn=None):
    rgn = np.random.RandomState() if rgn is None else rgn
    random_functions = [
        lambda x: np.log(x - np.min(x) + 1e-3),
        lambda x: np.sqrt(x - np.min(x)),
        lambda x: (1 + rgn.normal(size=x.shape)) * x,
        lambda x: x + rgn.normal(0, np.abs(np.mean(x)), size=x.shape)
    ]
    funcs = rgn.choice(random_functions, inputs.shape[1])
    return np.array([func(inputs[:, i]) for i, func in enumerate(funcs)]).T


def random_subset(inputs, targets, max_datasize=MAX_DATASIZE, rgn=np.random.RandomState()):

    classes_weights = random_classes_selection_and_weighting(targets, NB_FOLD, rgn)
    if classes_weights is None:
        return np.array([]), np.array([])
    idx_samples = random_samples_selection(targets, classes_weights, NB_FOLD, rgn=rgn, max_datasize=max_datasize)
    if idx_samples is None:
        return np.array([]), np.array([])
    idx_features = random_features_selection(inputs, rgn)
    inputs_ = inputs[idx_samples, :][:, idx_features]
    targets_ = targets[idx_samples]
    inputs_ = random_features_transformation(inputs_, rgn)
    return inputs_, targets_


def create_hpsearch_datasets(arff_filename):
    rgn = np.random.RandomState(42)
    output_prefix = os.path.join(UCI_CROSSVAL_FOLDER, arff_filename[:-5])
    inputs, targets = load_arff_dataset(os.path.join(UCI_FOLDER, arff_filename))
    nsamples, nfeatures = inputs.shape
    nclasses = len(np.unique(targets))
    ndatasets = int(np.ceil(np.log2((nsamples**2) * (nfeatures**2) * (nclasses**2))))
    print(dict(nsamples=nsamples, nfeatures=nfeatures, nclasses=nclasses, ndatasets=ndatasets))
    print(Counter(targets))
    if nsamples > 100:
        i = 0
        while i < ndatasets:
            print('generating dataset', i)
            x, y = random_subset(inputs, targets, MAX_DATASIZE, rgn)
            try:
                caracterize_learn_and_save_results(x, y, output_prefix + str(i))
                i += 1
            except Exception as err:
                print(err)
    else:
        caracterize_learn_and_save_results(inputs, targets, output_prefix)


if __name__ == '__main__':
    normalize_task_descr()
    exit()
    # for filename in os.listdir(UCI_CROSSVAL_FOLDER):
    #     filename = os.path.join(UCI_CROSSVAL_FOLDER, filename)
    #     with open(filename) as f:
    #         nb_lines = len(f.readlines())
    #     if nb_lines <= 5:
    #         os.remove(filename)
    # exit()
    import argparse, subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--server', default='local', type=str,
                        help='The name of the server: local|graham|helios')
    parser.add_argument('-p', '--part', default=0, type=int,
                        help='the id of the job that we should run in this execution')
    parser.add_argument('-m', '--pooling_mode', default='run', type=str,
                        help='How to run this script: dispatch jobs or run them')
    args = parser.parse_args()

    server, part, mode = args.server, args.part, args.mode
    # n_per_part = 20

    if mode == 'dispatch':
        filenames = [f for f in os.listdir(UCI_FOLDER) if f.endswith('arff')]
        ntasks = len(filenames)
        print("Quick summary")
        print('System:', server)
        print('Number of tasks dispatched:', ntasks)

        if server == "local":
            filenames = sorted([f for f in os.listdir(UCI_FOLDER) if f.endswith('arff')])
            for part in np.arange(len(filenames)):
                create_hpsearch_datasets(filenames[part])
        else:
            if server == 'graham':
                launcher, prototype = 'sbatch', "submit_graham.sh"
            elif server == 'helios':
                launcher, prototype = 'msub', "submit_helios.sh"
            else:
                launcher, prototype = None, None
                Exception("Server {} is not found".format(server))

            with open(prototype) as f:
                content = f.read()
                content = content.format(ntasks=ntasks)
            with open(prototype, 'w') as f:
                f.write(content)
            subprocess.Popen([launcher, prototype])
    else:
        if server == 'local':
            print('here')
            for f in os.listdir(UCI_FOLDER):
                if f.endswith('arff'):
                    create_hpsearch_datasets(f)
        else:
            filenames = sorted([f for f in os.listdir(UCI_FOLDER) if f.endswith('arff')])
            create_hpsearch_datasets(filenames[part])