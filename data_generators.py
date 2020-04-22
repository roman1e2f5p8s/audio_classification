from os.path import join 
import h5py
import numpy as np
import logging
from time import time


class DataGenerator:
    '''
    DataGenerator class
    '''
    def __init__(self, params, validate=False,
            time_steps=1): 
        '''
        Initialisation
        Arguments:
            - params -- parameters, hparams.HParamsFromYAML
            - validate -- whether split the dataset into train and validation sets, bool.
                Defaults to False
        '''
        self.hdf5_path = join(params.storage_dir, params.features_dir, 'log_mel', 'train.h5')
        # self.batch_size = params.batch_size
        # self.classes_number = params.classes_number
        self.validate = validate
        # self.seed = params.seed

        # TODO
        self.random_state = np.random.RandomState(params.seed)
        self.validate_random_state = np.random.RandomState(0)
        self.time_steps = time_steps
        self.hop_frames = self.time_steps // 2

        # load train.h5 file
        start_time = time()
        logging.info('Loading data from \"train.h5\"...')
        hdf5 = h5py.File(self.hdf5_path, 'r')

        labels = hdf5['label'][:]
        filenames = hdf5['filename'][:]

        self.labels = sorted({s.decode() for s in labels})  # TODO - try replacing sorted with list
        self.label_index_dict = {key: i for i, key in enumerate(self.labels)}
        self.filenames = [s.decode() for s in filenames]  # TODO - put np.array if required
        self.x = hdf5['feature'][:]
        self.y = np.array([self.label_index_dict[s.decode()] for s in labels])
        self.manually_verified = hdf5['manually_verified'][:]
        # st = time()
        # print(time() - st)

        hdf5.close()
        logging.info('Loading completed successfully. '
                'Elapsed time: {:.6f} s'.format(time() - start_time))

        if validate:
            self.validation_meta_file_path = join(params.storage_dir, params.validation_dir,
                params.validation_meta_file)
            self.train_audio_indexes, self.validate_audio_indexes = None, None
        else:
            self.train_audio_indexes = np.arange(len(filenames))
            self.validate_audio_indexes = np.empty(0)
        # print(self.train_audio_indexes)
        # print(self.validate_audio_indexes)

        def _train_validation_indices(self):
            pass



# d = DataGenerator('Storage/features/log_mel/train.h5', 1, 41, 1, 1)
