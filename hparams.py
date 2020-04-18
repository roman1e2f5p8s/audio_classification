from ruamel.yaml import YAML


class HParamsFromYAML:
    '''
    Loads hyperparameters from yaml file into HParamsFromYAML object
    '''
    def __init__(self, yaml_file, param_set='default'):
        '''
        Initialisation
        Arguments:
            - yaml_file -- name of yaml file, str
            - param_set -- set of parameters, str. Default value is 'default'
        '''
        self.yaml_file = yaml_file
        self.param_set = param_set
        with open(yaml_file) as f:
            self.model_params = YAML(typ='safe').load(f)[param_set]
        for name in self.model_params:
            setattr(self, name, self.model_params[name])
