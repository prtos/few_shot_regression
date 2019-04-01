import json
from loader import load_data
from model import BaseModel, ActiveModel
n_repeats = 30
max_samples = None


SAVING_DIR_FORMAT = '{expts_dir}/results_{dataset_name}_{algo}_{arch}'


def run_experiment(model_name, data_filename, train_size, y_scaler, output_path, input_path=None):
    x, y = load_data(data_filename, max_samples, y_dtype='float')
    data_name = data_filename.split('_')[-1].split('.')[0]
    param_grid = dict(n_estimators=[400], n_jobs=[-1], verbose=[1])
    if model_name in ['morgan_circular', 'meta']:
        if train_size > len(y) * 0.8:
            train_size = int(len(y) * 0.8)
        model = BaseModel(model_name='rf_rgr', fp_name=model_name,
                          tag=model_name, multitask=False)
        res = model.fit_eval(x, y, param_grid=param_grid,
                             train_size=train_size,
                             y_scaler=y_scaler,
                             cv=3,
                             n_jobs=-1,
                             verbose=1,
                             repeats=n_repeats,
                             batch_size=256)
    elif model_name == 'active':
        test_size = int(len(y) * 0.2)
        model = ActiveModel(model_name='rf_rgr', fp_name='morgan_circular', tag='rf', multitask=False)
        res = model.fit_eval(x, y, param_grid=param_grid,
                             test_size=test_size,
                             initial_train_size=10,
                             n_queries=train_size - 10,
                             y_scaler=y_scaler,
                             cv=3,
                             n_jobs=-1,
                             verbose=1,
                             repeats=n_repeats,
                             batch_size=256)
    else:
        raise Exception('model name incorrect')
    res.update(dict(data_name=data_name, model_name=model_name))
    print(res)

    with open("{}/{}_{}_{}_{}_res.json".format(output_path, data_name,
                                               model_name, train_size, y_scaler), 'w') as fd:
        json.dump(res, fd, indent=4, sort_keys=True)
