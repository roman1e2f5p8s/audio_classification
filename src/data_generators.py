from os.path import join 
import h5py
import numpy as np
from pandas import read_csv
import logging
from time import time


class DataGenerator:
    '''
    DataGenerator class
    '''
    def __init__(self, params, validate=False): 
        '''
        Initialisation
        Arguments:
            - params -- parameters, hparams.HParamsFromYAML
            - validate -- whether split the dataset into train and validation sets, bool.
                Defaults to False
        '''
        self.hdf5_path = join(params.storage_dir, params.features_dir, params.features_type, 'train.h5')
        self.input_frames_number = params.input_frames_number
        self.validate = validate
        self.batch_size = params.batch_size
        self.eval_audios_number = params.eval_audios_number
        # self.seed = params.seed

        self.train_rand_generator = np.random.RandomState(seed=params.seed)
        # TODO - why not to use params.seed?
        self.validation_rand_generator = np.random.RandomState(seed=0)

        # load train.h5 file
        start_time = time()
        logging.info('Loading data from \"train.h5\"...')
        hdf5 = h5py.File(self.hdf5_path, 'r')

        labels = hdf5['label'][:]
        filenames = hdf5['filename'][:]

        self.labels = sorted({s.decode() for s in labels})  # TODO - try replacing sorted with list,
        # or load labels from josn file
        self.label_index_dict = {key: i for i, key in enumerate(self.labels)}
        self.filenames = [s.decode() for s in filenames]  # TODO - put np.array if required
        self.x = hdf5['feature'][:]  # features
        self.y = np.array([self.label_index_dict[s.decode()] for s in labels])
        self.manually_verified = hdf5['manually_verified'][:]
        self.begin_end_indices = hdf5['begin_end_ind'][:]
        self.files_number = len(filenames)

        hdf5.close()
        logging.info('Loading completed successfully. '
                'Elapsed time: {:.6f} s'.format(time() - start_time))

        if validate:
            logging.info('Generating train and validation indices for audio files...')
            folds = read_csv(join(params.storage_dir, params.validation_dir,
                    params.validation_meta_file), usecols=['fold'])['fold']
            self.train_audio_indices = folds.index[folds != params.holdout_fold].to_numpy()
            self.validation_audio_indices = folds.index[folds == params.holdout_fold].to_numpy()
            logging.info('Generating completed successfully')
        else:
            self.train_audio_indices = np.arange(self.files_number)
            self.validation_audio_indices = np.empty(0)
        self.train_audio_indices_len = len(self.train_audio_indices)
        self.validation_audio_indices_len = len(self.validation_audio_indices)

        print('|----------------------------------------------------------------------------')
        logging.info('Number of audio files for training: {}'.format(self.train_audio_indices_len))
        logging.info('Number of audio files for validation: {}'.\
                format(self.validation_audio_indices_len))
        if validate:
            logging.info('Train-validation split ratio is approximately {:.6f}:1'.\
                    format(self.train_audio_indices_len / self.validation_audio_indices_len))

        train_begin_end_indices = self.begin_end_indices[self.train_audio_indices]
        x_train = [self.x[begin:end] for [begin, end] in train_begin_end_indices]

        # join along axis 0 such that the resulting array will have shape (*, mels_number)
        x_train = np.concatenate(x_train, axis=0)
        if x_train.ndim == 3:  # TODO - remove it
            exit('ndim of x_train = 3 but not 2!! change axis to (0, 1) instead of 0 in mean/std')

        # compute mean and std for training data, the resulting arrays will have shape (mels_number,)
        self.mean = np.mean(x_train, axis=0)
        self.std = np.std(x_train, axis=0)

        # generate training chunks
        self.train_chunks = []
        for i in range(self.train_audio_indices_len):
            [begin, end] = train_begin_end_indices[i]
            audio_label_ind = self.y[self.train_audio_indices[i]]
            self.train_chunks += \
                self.__generate_chunks(begin, end, audio_label_ind)
        self.train_chunks_len = len(self.train_chunks)
        self.epoch_len = self.train_chunks_len // self.batch_size + 1
        self.epoch = 1
        
        logging.info('Number of chunks for training: {}'.format(self.train_chunks_len))
        logging.info('Batch size: {}'.format(self.batch_size))
        logging.info('One epoch lasts {} iterations'.format(self.epoch_len))
        print('|----------------------------------------------------------------------------')

    def __generate_chunks(self, begin, end, audio_label_ind):
        '''
        Method that splits log-mel frames if the number of frames is > than self.input_frames_number.
        Parameters:
            - begin -- index that indicates the beginning of audiodata in the whole bunch, int >=0
            - end -- index that indicates the end of audiodata in the whole bunch, int >=0
            - audio_label_ind -- index of audio's label int >=0, <= number of classes
        Returns:
            - list of tuples (BEGIN, END, audio_label_ind)
        '''
        if end - begin <= self.input_frames_number:
            return [(begin, end, audio_label_ind)]
        else:
            begin_indices = np.arange(begin, end - self.input_frames_number,
                    step=(self.input_frames_number // 2))
            return [(b, b + self.input_frames_number, audio_label_ind) for b in begin_indices]

    def train_generator(self):
        '''
        Generates batches for training using generator object and yield
        Parameters:
            - None
        Returns:
            - generator object that yields a tuple of x and y batches both being numpy arrays
        '''
        train_chunks_copy = self.train_chunks.copy()
        self.train_rand_generator.shuffle(train_chunks_copy)

        batch_begin_ind = 0
        while True:
            # set batch_begin_ind to 0 and reshuffled data at every epoch
            if batch_begin_ind >= self.train_chunks_len:
                batch_begin_ind = 0
                self.train_rand_generator.shuffle(train_chunks_copy)
                self.epoch += 1
            batch_indices = train_chunks_copy[batch_begin_ind:batch_begin_ind+self.batch_size]

            x_batch, y_batch = [], []
            for (begin, end, y) in batch_indices:
                y_batch += [y]
                x = self.x[begin:end] 
                x_batch += [x] if end - begin == self.input_frames_number else \
                        [np.tile(x, (self.input_frames_number//len(x)+1, 1))[0:self.input_frames_number]]
                        # repeat if the number of frames is smaller than input_frames_number
            # scale x data
            x_batch = (np.array(x_batch) - self.mean) / self.std

            batch_begin_ind += self.batch_size

            yield x_batch, np.array(y_batch)

    def validation_generator(self, validate, manually_verified_only, shuffle):
        '''
        Parameters:
            - validate -- if true, use the validation dataset, else use self.eval_audios_number
                training data to evaluate the model, bool
            - manually_verified_only -- use only manually verified audios for evaluation, bool
            - shuffle -- shuffle or not the validation data, bool
        Returns:
            - generator object that yields a tuple of x, y batches both being numpy arrays,
                label y (int) and audio filename being a string
        ''' 
        audio_indices = self.validation_audio_indices if validate else self.train_audio_indices
        if manually_verified_only:
            audio_indices = audio_indices[np.where(self.manually_verified[audio_indices] == 1)[0]]
        if shuffle:
            self.validation_rand_generator.shuffle(audio_indices)
        for (ind, aind) in enumerate(audio_indices):
            if ind == self.eval_audios_number:
                break
            [begin, end] = self.begin_end_indices[aind]
            y = self.y[aind]
            chunk_indices = self.__generate_chunks(begin, end, y)
            x_batch, y_batch = [], []
            for (begin, end, y_) in chunk_indices:
                y_batch += [y_]
                x = self.x[begin:end] 
                x_batch += [x] if end - begin == self.input_frames_number else \
                        [np.tile(x, (self.input_frames_number//len(x)+1, 1))[0:self.input_frames_number]]
                        # repeat if the number of frames is smaller than input_frames_number
            # scale x data
            x_batch = (np.array(x_batch) - self.mean) / self.std

            yield x_batch, np.array(y_batch), y, self.filenames[aind]
