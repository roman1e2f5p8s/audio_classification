from src.parser import Parser
from src.hparams import HParamsFromYAML
from src.plot_utils import plot_train, plot_validation_test


def main():
    '''
    Main function
    Arguments:
        - None
    Returns:
        - None
    '''
    # create parser of arguments
    args = Parser(parser_mode='plot').args

    if args.verbose:
        print('|------------------------------------------------------------------------------')

    # get hyperparameters
    params = HParamsFromYAML('hparams.yaml', param_sets=[args.features, args.model])

    if args.mode == 'train':
        if args.verbose:
            print('|Plotting results of training...')
        plot_train(
                params=params,
                validated=args.validated,
                manually_verified_only=args.manually_verified_only,
                latex=args.latex,
                verbose=args.verbose
                )
    elif args.mode == 'validation':
        if args.verbose:
            print('|Plotting results of training...')
        plot_validation_test(
                params=params,
                checkpoint_epoch=args.epoch,
                validated=args.validated,
                manually_verified_only=args.manually_verified_only,
                latex=args.latex,
                verbose=args.verbose,
                mode=args.mode
                )
    elif args.mode == 'test':
        if args.verbose:
            print('|Plotting results of test...')
        plot_validation_test(
                params=params,
                checkpoint_epoch=args.epoch,
                validated=args.validated,
                manually_verified_only=args.manually_verified_only,
                latex=args.latex,
                verbose=args.verbose,
                mode=args.mode
                )

    if args.verbose:
        print('|Plotting completed successfully')
        print('|------------------------------------------------------------------------------')


if __name__ == '__main__':
    main()

