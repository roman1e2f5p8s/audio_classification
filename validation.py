import os
import pandas as pd
# from pandas import to_csv
# from json import load  # TODO - remove it
from time import time
import logging

import numpy as np


def generate_validation_metadata(params, folds_number=4, seed=1234):
    '''
    Generates a validation dataset, creates a validation file 
    Parameters:
        - params -- parameters, hparams.HParamsFromYAML
        - folds_number -- number of folds, int >= 2. Defaults to 4
        - seed -- seed used to initialise the pseudo-random number generator, int >= 0. Defaults to 1234
    Returns:
        - None
    '''
    # params.datasets_dir
    # params.storage_dir

    logging.info('Generating validation metadata...')

    if folds_number < 2:
        logging.warning('Number of folds must be at least 2!\n\t'
                'Default value of 4 has been used instead of {}'.format(folds_number))

    # with open(join(params.storage_dir, params.labels_file), 'r') as f:  # TODO - remove it
        # LABELS = load(f)['labels']  # TODO - remove it

    rand_generator = np.random.RandomState(seed=seed)

    META_FILE_PATH = os.path.join(params.datasets_dir, params.metadata_dir, params.train_meta_file)
    metadata_df = pd.read_csv(META_FILE_PATH)   # TODO: put pd.DataFrame() on read_csv if required
    metadata_df_len = len(metadata_df)
    # st = time()
    # print(time() - st)

    indices = np.arange(metadata_df_len)
    rand_generator.shuffle(indices)

    folds = np.zeros(metadata_df_len, dtype=np.int32)
    for n in range(metadata_df_len):
        folds[indices[n]] = (n % folds_number) + 1
    metadata_df['fold'] = folds
    VALIDATION_META_FILE_DIR = os.path.join(params.storage_dir, params.validation_dir)
    if not os.path.exists(VALIDATION_META_FILE_DIR):
        os.makedirs(VALIDATION_META_FILE_DIR)
    metadata_df.to_csv(os.path.join(VALIDATION_META_FILE_DIR, params.validation_meta_file))

    logging.info('Generating completed successfully.')
