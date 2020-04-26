import argparse


class Parser:
    def __init__(self):
        # create the main parser and subparsers
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(
                dest='mode',
                title='mode',
                description='valid modes:',
                required=True,
                help='Call python main.py {train,validation,test} -h[--help] to see help for a given mode'
                )
        
        # parser for training
        parser_train = subparsers.add_parser('train', help='b')
        # required arguments in the train mode
        parser_train_req = parser_train.add_argument_group('required arguments')
        parser_train_req.add_argument(
                '--model',
                type=str,
                choices=['VGGish', 'CNN'],
                required=True,
                help='One of available models'
                )
        parser_train_req.add_argument(
                '--features',
                type=str,
                choices=['log_mel', 'mfcc', 'raw'],
                required=True,
                help='One of available methods to extract features'
                )
        # optional arguments in the train mode
        parser_train_opt = parser_train.add_argument_group('optional arguments')
        parser_train_opt.add_argument(
                '--validate',
                action='store_true',
                default=True,
                required=False,
                help='Evaluate or not the model during the training on a separate '
                    'validation dataset. If False, a subset of training data will be used for evaluation. '
                    'Defaults to True'
                )
        parser_train_opt.add_argument(
                '--manually_verified_only',
                action='store_true',
                default=True,
                required=False,
                help='If true, evaluate the model using only manually verified data. Defaults to True'
                )
        parser_train_opt.add_argument(
                '--shuffle',
                action='store_true',
                default=True,
                required=False,
                help='Shuffle or not the data for evaluation. Defaults to True'
                )
        parser_train_opt.add_argument(
                '--cuda',
                action='store_true',
                default=False,
                required=False,
                help='Use CUDA (if available) or not. Defaults to False')
        parser_train_opt.add_argument(
                '-v', '--verbose',
                action='store_true',
                default=True,
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
                help='One of available models'
                )
        parser_validation_req.add_argument(
                '--features',
                type=str,
                choices=['log_mel', 'mfcc', 'raw'],
                required=True,
                help='One of available methods to extract features'
                )
        parser_validation_req.add_argument(
                '--epoch',
                type=int,
                required=True,
                help='From which epoch use model for validation'
                )
        # optional arguments in the validation mode
        parser_validation_opt = parser_validation.add_argument_group('optional arguments')
        parser_validation_opt.add_argument(
                '--manually_verified_only',
                action='store_true',
                default=True,
                required=False,
                help='If true, validate the model using only manually verified data. Defaults to True'
                )
        parser_validation_opt.add_argument(
                '--shuffle',
                action='store_true',
                default=False,
                required=False,
                help='Shuffle or not the data for evaluation. Defaults to False'
                )
        parser_validation_opt.add_argument(
                '--cuda',
                action='store_true',
                default=False,
                required=False,
                help='Use CUDA (if available) or not. Defaults to False'
                )
        parser_validation_opt.add_argument(
                '-v', '--verbose',
                action='store_true',
                default=True,
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
                help='One of available models'
                )
        parser_test_req.add_argument(
                '--features',
                type=str,
                choices=['log_mel', 'mfcc', 'raw'],
                required=True,
                help='One of available methods to extract features'
                )
        parser_test_req.add_argument(
                '--epoch',
                type=int,
                required=True,
                help='From which epoch use model for testing'
                )
        # optional arguments in the validation mode
        parser_test_opt = parser_test.add_argument_group('optional arguments')
        parser_test_opt.add_argument(
                '--cuda',
                action='store_true',
                default=False,
                required=False,
                help='Use CUDA (if available) or not. Defaults to False'
                )
        parser_test_opt.add_argument(
                '-v', '--verbose',
                action='store_true',
                default=True,
                required=False,
                help='Print or not the to console. Defaults to True'
                )

        self.args = parser.parse_args()
