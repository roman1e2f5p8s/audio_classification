import os
import logging
import numpy as np
import pandas as pd


def generate_validation_metadata(params):
    '''
    Generates validation meta file 
    Parameters:
        - params -- parameters, hparams.HParamsFromYAML
    Returns:
        - None
    '''

    logging.info('Generating validation metadata...')

    rand_generator = np.random.RandomState(seed=params.seed)

    META_FILE_PATH = os.path.join(params.datasets_dir, params.metadata_dir, params.train_meta_file)
    metadata_df = pd.read_csv(META_FILE_PATH)
    metadata_df_len = len(metadata_df)

    indices = np.arange(metadata_df_len)
    rand_generator.shuffle(indices)

    folds = np.zeros(metadata_df_len, dtype=np.int32)
    for i in range(metadata_df_len):
        folds[indices[i]] = (i % params.folds_number) + 1
    metadata_df['fold'] = folds

    VALIDATION_META_FILE_DIR = os.path.join(params.storage_dir, params.validation_dir)
    if not os.path.exists(VALIDATION_META_FILE_DIR):
        os.makedirs(VALIDATION_META_FILE_DIR)
    metadata_df.to_csv(os.path.join(VALIDATION_META_FILE_DIR, params.validation_meta_file))

    logging.info('Generating completed successfully.')
