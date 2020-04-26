import os
from json import dump
from pandas import read_csv
import logging


def extract_labels(params):
    '''
    Extracts labels from a train.csv file. Labels will be stored in a json file
    Parameters:
        - params -- parameters, hparams.HParamsFromYAML
    Returns:
        - None
    '''
    logging.info('Extracting labels for audios...')

    if not os.path.exists(params.storage_dir):
        os.makedirs(params.storage_dir)
    LABELS_JSON_PATH = os.path.join(params.storage_dir, params.labels_file)

    META_FILE_PATH = os.path.join(params.datasets_dir, params.metadata_dir, params.train_meta_file)
    LABELS = read_csv(META_FILE_PATH, usecols=['label'])['label'].unique()

    with open(LABELS_JSON_PATH, 'w') as f:
        dump({'labels': sorted(LABELS)}, f, indent=4)

    logging.info('Extracting completed successfully.\n|Labels saved in \"{}\"'.format(LABELS_JSON_PATH))
