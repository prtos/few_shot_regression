from metalearn.models.factory import ModelFactory


class ConfigFactory:
    def __init__(self):
        super(ConfigFactory, self).__init__()

    def __call__(self, dataset_name):
        if dataset_name in ['toy', 'easytoy', 'toy123']:
            import config_toy as cfg
        else:
            raise Exception("Dataset {} is not found".format(dataset_name))

        cfg_content = vars(cfg)

        algo_params = {
            var_name: cfg_content[var_name]
            for var_name in cfg_content
            if var_name in list(ModelFactory.name_map.keys())
        }

        if dataset_name in ['toy', 'easytoy', 'toy123']:
            for algo_name in algo_params:
                if isinstance(algo_params[algo_name], dict):
                    algo_params[algo_name].update(dict(dataset_name=[dataset_name]))
                elif isinstance(algo_params[algo_name], list):
                    for sub_dict in algo_params[algo_name]:
                        sub_dict.update(dict(dataset_name=[dataset_name]))
                else:
                    raise Exception("Something wrong happens here. algo_param is neither a list nor a dict.")
        return algo_params


if __name__ == '__main__':
    print(ConfigFactory()('toy')['metakrr_sk']['dataset_name'])
    print(ConfigFactory()('easytoy')['metakrr_sk']['dataset_name'])
    print(ConfigFactory()('toy123')['metakrr_sk']['dataset_name'])
