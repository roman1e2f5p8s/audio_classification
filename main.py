import os
from time import time
from importlib import import_module

from src.parser import Parser
from src.hparams import HParamsFromYAML
from src.logger import Logger
from src.main_utils import reproducibility, get_device, count_model_params, train, validate, test


def main():
    '''
    Main function
    Arguments:
        - None
    Returns:
        - None
    '''
    START_TIME = time()

    # create parser of arguments
    args = Parser(parser_mode='main').args

    # get hyper-parameters
    params = HParamsFromYAML('hparams.yaml', param_sets=[args.features, args.model])

    LOGS_DIR = os.path.join(params.logs_dir, args.model, args.features, args.mode) 
    logg = Logger(LOGS_DIR, verbose=args.verbose)
    logger = logg.logger

    if args.verbose:
        print('|-------------------------------------------------------------------------------')
    logger.info('===== Model: {}, features: {}. Mode: {} ====='.format(args.model, args.features,
            args.mode))
    if args.verbose:
        print('|-------------------------------------------------------------------------------')

    # extract labels if labels.json does not exist yet
    if not os.path.isfile(os.path.join(params.storage_dir, params.labels_file)):
        from src.extract_labels import extract_labels
        extract_labels(params)

    # generate validation metadata if validation_meta.csv does not exist yet
    if (args.mode == 'train' and args.validate) or args.mode == 'validation':
        if not os.path.isfile(os.path.join(params.storage_dir, params.validation_dir,
                params.validation_meta_file)):
            from src.validation import generate_validation_metadata
            generate_validation_metadata(params)

    if args.mode == 'train' or args.mode == 'test':
        # extract features if the corresponding h5 file does not exist yet
        if not os.path.isfile(os.path.join(params.storage_dir, params.features_dir, args.features,
                '{}.h5'.format(args.mode))):
            features_module = import_module('src.{}_features'.format(args.features))
            features_module.extract_features(args.mode)

    # for reproducibility
    reproducibility(seed=params.seed)

    # get model
    models = import_module('src.models')
    model = getattr(models, args.model)(params)
    logger.info('Number of trainable parameters: {}'.format(count_model_params(model)))

    # get device
    device = get_device(cuda_flag=args.cuda)
    if args.verbose:
        print('|-------------------------------------------------------------------------------')

    if args.mode == 'train':
        train(
                params=params,
                model=model,
                device=device,
                validate=args.validate,
                manually_verified_only=args.manually_verified_only,
                shuffle=args.shuffle,
                verbose=args.verbose
                )
    elif args.mode == 'validation':
        validate(
                params=params,
                model=model,
                device=device,
                checkpoint_epoch=args.epoch,
                validated=args.validated,
                manually_verified_only=args.manually_verified_only,
                shuffle=args.shuffle,
                verbose=args.verbose
                )
    elif args.mode == 'test':
        test(
                params=params,
                model=model,
                device=device,
                checkpoint_epoch=args.epoch,
                validated=args.validated,
                manually_verified_only=args.manually_verified_only,
                verbose=args.verbose
                )
    logger.info('Total time: {:.6f} s'.format(time() - START_TIME))
    if args.verbose:
        print('|-------------------------------------------------------------------------------')


if __name__ == '__main__':
    main()
