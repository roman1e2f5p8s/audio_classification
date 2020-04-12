from ruamel.yaml import YAML


class HParamsFromYAML:
    '''
    Loads hyperparameters from yaml file into HParamsFromYAML object
    '''
    def __init__(self, yaml_file, model_name):
        '''
        Initialisation
        Arguments:
            - yaml_file -- name of yaml file, str
            - model_name -- name of model, str
        '''
        self.yaml_file = yaml_file
        self.model_name = model_name
        with open(yaml_file) as f:
            self.model_params = YAML(typ='safe').load(f)[model_name]
        for name in self.model_params:
            setattr(self, name, self.model_params[name])
