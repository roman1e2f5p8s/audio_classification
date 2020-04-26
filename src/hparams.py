from ruamel.yaml import YAML


class HParamsFromYAML:
    '''
    Loads hyperparameters from yaml file into HParamsFromYAML object
    '''
    def __init__(self, yaml_file, param_sets=['default']):
        '''
        Initialisation
        Arguments:
            - yaml_file -- name of yaml file, str
            - param_sets -- sets of parameters, list. Defaults to ['default']
        '''
        if len(param_sets) > 2:
            exit('No more than two parameter sets are allowed! Exiting.')
        self.yaml_file = yaml_file
        self.param_sets = param_sets
        self.params = {}
        with open(yaml_file) as f:
            yaml = YAML(typ='safe').load(f)
            for param_set in param_sets:
                params_dict = yaml[param_set]
                if not self.params:
                    self.params = params_dict
                else:
                    for key in params_dict:
                        self.params[key] = params_dict[key]
        for name in self.params:
            setattr(self, name, self.params[name])
