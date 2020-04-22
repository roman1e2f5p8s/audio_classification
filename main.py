import os
import h5py
# import logging
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from hparams import HParamsFromYAML
from logger import Logger
from models import VGGish
from data_generators import DataGenerator

args_validate = True
args_filename = __file__.split('.')[0]
args_holdout_fold = 1
args_model_name = 'VGGish'
args_cuda = False
args_cpu = False


def train(params, validate, filename, holdout_fold):
    '''
    Parameters:
        - params -- model parameters, hparams.HParamsFromYAML
        - validate -- whether split the dataset into train and validation sets
        - filename -- name of this script withut extension, e.g. main, str
        - holdout_fold -- int
    '''
    # get some parameters from yaml file  # TODO - delete it
    # params = HParamsFromYAML('hparams.yaml', param_set=args_model_name)  # TODO - delete it

    # path to train.h5 file with features
    HDF5_PATH = os.path.join(params.storage_dir, 'features', 'log_mel', 'train.h5')

    if validate:
        MODELS_DIR = os.path.join(params.storage_dir, 'models', filename,
                'holdout_fold{}'.format(holdout_fold))
    else:
        MODELS_DIR = os.path.join(params.storage_dir, 'models', filename, 'train')
        # 'train' == full_train
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    model = VGGish(
            classes_number=params.classes_number,
            conv_blocks_number=params.conv_blocks_number,
            kernel_size=(params.kernel_size_1st_dim, params.kernel_size_2nd_dim),
            stride=(params.stride_1st_dim, params.stride_2nd_dim),
            padding=(params.padding_1st_dim, params.padding_2nd_dim),
            add_bias_to_blocks=params.add_bias_to_blocks,
            init_layers_manually=params.init_layers_manually,
            std_scale_factor=params.std_scale_factor,
            max_pool_kernel_size=(params.max_pool_kernel_size_1st_dim,
                params.max_pool_kernel_size_2nd_dim),
            max_pool_stride=(params.max_pool_stride_1st_dim, params.max_pool_stride_2nd_dim),
            add_bias_to_fc=params.add_bias_to_fc)

    if args_cuda:
        # TODO - check availability
        model.cuda()
    else:
        if args_cpu:
            model.cpu()

    optimiser = optim.Adam(
            params=model.parameters(),
            lr=params.learning_rate,
            betas=(params.beta_1, params.beta_2),
            eps=params.eps,
            weight_decay=params.weight_decay,
            amsgrad=params.amsgrad)

    data_generator = DataGenerator(params, validate=validate)


def main():
    '''
    '''
    # print(args_filename)
    params = HParamsFromYAML('hparams.yaml', param_set=args_model_name)

    logs_dir = os.path.join(params.logs_dir, args_model_name, args_filename) 
    logger = Logger(logs_dir).logger
    # logger.info(params)
    # logging.info(params)
    # st = time()
    # print(time() - st)

    # extract labels if labels.json does not exist
    if not os.path.isfile(os.path.join(params.storage_dir, params.labels_file)):
        from extract_labels import extract_labels
        extract_labels(params)

    if args_validate:
        # generate validation if validation_meta.csv does not exist
        if not os.path.isfile(os.path.join(params.storage_dir, params.validation_dir,
                params.validation_meta_file)):
            from validation import generate_validation_metadata
            generate_validation_metadata(params)

    train(params, args_validate, args_filename, args_holdout_fold)


if __name__ == '__main__':
    '''
    '''
    main()
