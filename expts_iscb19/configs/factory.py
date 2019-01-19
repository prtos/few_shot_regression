class ConfigFactory:
    def __init__(self):
        super(ConfigFactory, self).__init__()
        
    def __call__(self, dataset_name):
        if dataset_name == 'mhc':
            import configs.config_mhc as cfg
        elif dataset_name == 'chembl':
            import configs.config_chembl as cfg
        elif dataset_name == 'pubchemtox':
            import configs.config_pubchemtox as cfg
        elif dataset_name == 'tox21':
            import configs.config_tox21 as cfg
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
