class ConfigFactory:
    def __init__(self):
        super(ConfigFactory, self).__init__()
        
    def __call__(self, dataset_name):
        if dataset_name == 'mhc':
            import configs.config_mhc as cfg
        elif dataset_name == 'bindingdb':
            import configs.config_bdb as cfg
        elif dataset_name == 'movielens':
            import configs.config_movielens as cfg
        elif dataset_name == 'uci':
            import configs.config_uci as cfg
        elif dataset_name == 'toy' or dataset_name == 'easytoy':
            import configs.config_toy as cfg
        else:
            raise Exception("Dataset {} is not found".format(dataset_name))

        cfg_content = vars(cfg)
        algo_params = {
            cfg_content[var_name]['model_name'][0]: cfg_content[var_name] 
            for var_name in cfg_content 
            if isinstance(cfg_content[var_name], dict) and 
            'model_name' in cfg_content[var_name]
            }
        return algo_params
