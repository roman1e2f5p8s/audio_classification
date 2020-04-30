import argparse


class Parser:
    '''
    Parser class
    '''
    def __init__(self, parser_mode='main'):
        '''
        Initialisation
        Arguments:
            - parser_mode -- 'main' or 'plot', str. Defaults to 'main'
        '''
        if not parser_mode in ['main', 'plot']:
            exit('Wrong parser mode \'{}\'. Supported only \'main\' or \'plot\'. Exiting.'\
                    .format(parser_mode))
        self.parser_mode = parser_mode
        _plot = '. Plots results for this case' * (parser_mode == 'plot')
        _d = 'd' * (parser_mode == 'plot')

        # create the main parser and subparsers
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(
                dest='mode',
                title='mode',
                description='valid modes:',
                required=True,
                help='Call python {}.py {{train,validation,test}} -h[--help] '.format(parser_mode) +\
                    'to see help for a given mode'
                )
        
        # parser for training
        parser_train = subparsers.add_parser('train')
        # required arguments in the train mode
        parser_train_req = parser_train.add_argument_group('required arguments')
        parser_train_req.add_argument(
                '--model',
                type=str,
                choices=['VGGish', 'CNN'],
                required=True,
                help='One of available models{}'.format(_plot)
                )
        parser_train_req.add_argument(
                '--features',
                type=str,
                choices=['log_mel', 'mfcc', 'chroma'],
                required=True,
                help='One of available methods to extract features{}'.format(_plot)
                )
        # optional arguments in the train mode
        parser_train_opt = parser_train.add_argument_group('optional arguments')
        parser_train_opt.add_argument(
                '--validate{}'.format(_d),
                action='store_true',
                default=False,
                required=False,
                help='Evaluate or not the model during the training on a separate validation dataset. '
                    'If False, a subset of training data will be used for evaluation. '
                    'Defaults to True{}'.format(_plot)
                )
        parser_train_opt.add_argument(
                '--manually_verified_only',
                action='store_true',
                default=False,
                required=False,
                help='If true, evaluate the model using only manually verified data. '
                    'Defaults to True{}'.format(_plot)
                )
        if parser_mode == 'main':
            parser_train_opt.add_argument(
                    '--shuffle',
                    action='store_true',
                    default=False,
                    required=False,
                    help='Shuffle or not the data for evaluation. Defaults to True{}'.format(_plot)
                    )
            parser_train_opt.add_argument(
                    '--cuda',
                    action='store_true',
                    default=False,
                    required=False,
                    help='Use CUDA (if available) or not. Defaults to False')
        else:
            parser_train_opt.add_argument(
                    '--latex',
                    action='store_true',
                    default=False,
                    required=False,
                    help='Use LaTeX for plotting or not. Defaults to False'
                    )
        parser_train_opt.add_argument(
                '--verbose',
                action='store_true',
                default=False,
                required=False,
                help='Print or not the to console. Defaults to True'
                )

        # parser for validation
        parser_validation = subparsers.add_parser('validation')
        # required arguments in the validation mode
        parser_validation_req = parser_validation.add_argument_group('required arguments')
        parser_validation_req.add_argument(
                '--model',
                type=str,
                choices=['VGGish', 'CNN'],
                required=True,
                help='One of available models{}'.format(_plot)
                )
        parser_validation_req.add_argument(
                '--features',
                type=str,
                choices=['log_mel', 'mfcc', 'chroma'],
                required=True,
                help='One of available methods to extract features{}'.format(_plot)
                )
        parser_validation_req.add_argument(
                '--epoch',
                type=int,
                required=True,
                help='From which epoch use model for validation{}'.format(_plot)
                )
        # optional arguments in the validation mode
        parser_validation_opt = parser_validation.add_argument_group('optional arguments')
        parser_validation_opt.add_argument(
                '--validated',
                action='store_true',
                default=False,
                required=False,
                help='Use validated model or not. Defaults to True{}'.format(_plot)
                )
        parser_validation_opt.add_argument(
                '--manually_verified_only',
                action='store_true',
                default=False,
                required=False,
                help='If true, validate the model using only manually verified data. '
                    'Defaults to True{}'.format(_plot)
                )
        if parser_mode == 'main':
            parser_validation_opt.add_argument(
                    '--shuffle',
                    action='store_true',
                    default=False,
                    required=False,
                    help='Shuffle or not the data for evaluation. Defaults to False{}'.format(_plot)
                    )
            parser_validation_opt.add_argument(
                    '--cuda',
                    action='store_true',
                    default=False,
                    required=False,
                    help='Use CUDA (if available) or not. Defaults to False'
                    )
        else:
            parser_validation_opt.add_argument(
                    '--latex',
                    action='store_true',
                    default=False,
                    required=False,
                    help='Use LaTeX for plotting or not. Defaults to False'
                    )
        parser_validation_opt.add_argument(
                '--verbose',
                action='store_true',
                default=False,
                required=False,
                help='Print or not the to console. Defaults to True'
                )

        # parser for testing
        parser_test = subparsers.add_parser('test')
        # required arguments in the test mode
        parser_test_req = parser_test.add_argument_group('required arguments')
        parser_test_req.add_argument(
                '--model',
                type=str,
                choices=['VGGish', 'CNN'],
                required=True,
                help='One of available models{}'.format(_plot)
                )
        parser_test_req.add_argument(
                '--features',
                type=str,
                choices=['log_mel', 'mfcc', 'chroma'],
                required=True,
                help='One of available methods to extract features{}'.format(_plot)
                )
        parser_test_req.add_argument(
                '--epoch',
                type=int,
                required=True,
                help='From which epoch use model for testing{}'.format(_plot)
                )
        # optional arguments in the validation mode
        parser_test_opt = parser_test.add_argument_group('optional arguments')
        parser_test_opt.add_argument(
                '--validated',
                action='store_true',
                default=False,
                required=False,
                help='Use validated model or not. Defaults to True{}'.format(_plot)
                )
        parser_test_opt.add_argument(
                '--manually_verified_only',
                action='store_true',
                default=False,
                required=False,
                help='If True, use the model that was evaluated using only manually verified data. '
                    'Defaults to True{}'.format(_plot)
                )
        if parser_mode == 'main':
            parser_test_opt.add_argument(
                    '--cuda',
                    action='store_true',
                    default=False,
                    required=False,
                    help='Use CUDA (if available) or not. Defaults to False'
                    )
        else:
            parser_test_opt.add_argument(
                    '--latex',
                    action='store_true',
                    default=False,
                    required=False,
                    help='Use LaTeX for plotting or not. Defaults to False'
                    )
        parser_test_opt.add_argument(
                '--verbose',
                action='store_true',
                default=False,
                required=False,
                help='Print or not the to console. Defaults to True'
                )

        self.args = parser.parse_args()
