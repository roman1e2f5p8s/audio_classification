import os
import h5py

from hparams import HParamsFromYAML


def train(validate, filename, holdout_fold):
    '''
    Parameters:
        - validate -- bool
        - filename -- name of this script withut extension, e.g. main, str
        - holdout_fold -- int
    '''
    # get some parameters from yaml file
    params = HParamsFromYAML('hparams.yaml', param_set='default')

    # path to train.h5 file with features
    HDF5_PATH = os.path.join(params.storage_dir, 'features', 'log_mel', 'train.h5')

    if validate:
        VALIDATION_CSV_PATH = os.path.join(params.storage_dir, 'validate_meta.csv')
        MODELS_DIR = os.path.join(params.storage_dir, 'models', filename,
                'holdout_fold{}'.format(holdout_fold))  # holdout_fold -> validate
    else:
        VALIDATION_CSV_PATH = None
        MODELS_DIR = os.path.join(params.storage_dir, 'models', filename, 'train')
        # 'train' == full_train
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)


def main():
    '''
    '''
    validate = True
    filename = __file__.split('.')[0]
    holdout_fold = 1
    
    print(filename)
    train(validate, filename)


if __name__ == '__main__':
    '''
    '''
    main()
