import h5py
import numpy as np
import logging
from time import time


class DataGenerator:
    '''
    DataGenerator class
    '''
    def __init__(self, hdf5_path, batch_size, classes_number,
            time_steps, seed): 
        '''
        Initialisation
        Arguments:
            - hdf5_path -- path to train.h5 file with features, str
            - batch_size -- batch size: how many samples per batch to load, int
            - classes_number -- number of classes in a multiclass classfication problem, int > 0
        '''
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.classes_number = classes_number

        # TODO
        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)
        self.time_steps = time_steps
        self.hop_frames = self.time_steps // 2

        # load train.h5 file
        hdf5 = h5py.File(hdf5_path, 'r')

        labels = hdf5['label'][:]
        filenames = hdf5['filename'][:]

        self.labels = sorted({s.decode() for s in labels})  # TODO - try replacing sorted with list
        self.label_index_dict = {key: i for i, key in enumerate(self.labels)}
        self.filenames = [s.decode() for s in filenames]  # TODO - put np.array if required
        self.x = hdf5['feature'][:]
        self.y = np.array([self.label_index_dict[s.decode()] for s in labels])
        self.manually_verified = hdf5['manually_verified'][:]
        logging.info('Loading data time: {:.3f} s'.format(1))
        # st = time()
        # print(time() - st)

        hdf5.close()


d = DataGenerator('Storage/features/log_mel/train.h5', 1, 41, 1, 1)
