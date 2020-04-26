import os
import h5py
from time import time
import numpy as np
from pandas import read_csv
import librosa
import logging

from .hparams import HParamsFromYAML


class LogMelSpectrogram:
    '''
    Stores the common parameters and computes a log10 mel-scaled spectrogram.
    '''
    def __init__(self, params):
        '''
        Initialisation
        Arguments:
            - params -- model parameters, hparams.HParamsFromYAML
        Attributes:
            - sample_rate -- target sampling rate for feature extraction, int > 0
            - fft_window_size -- length of the FFT window, int > 0
            - mels_number -- number of Mel bands to generate, int > 0
            - hop_length -- number of samples between successive frames, int > 0
            - min_frequency -- lowest frequency (in Hz), float >= 0
            - max_frequency -- highest frequency (in Hz), float >= 0. If None, use sample_rate // 2
        '''
        self.sample_rate = params.sample_rate
        self.fft_window_size = params.fft_window_size
        self.mels_number = params.mels_number
        self.hop_length = params.hop_length
        self.min_frequency = params.min_frequency
        self.max_frequency = params.sample_rate // 2 if params.max_frequency is None \
                else params.max_frequency
        
    def get_spectrogram(self, path_to_audio):
        '''
        Reads and resamples audio using the librosa library, normalises the audiodata
        and returns log mel spectrogram using parameters obtained during the initialisation
        Parameters:
            - path_to_audio -- path to audio file, str
        Returns:
            - transposed log10 mel spectrogram, numpy.ndarray of shape
            (len(wavedata)//self.hop_length-1, self.mels_number)
        '''
        wavedata, _ = librosa.core.load(path_to_audio, sr=self.sample_rate)
        wavedata = wavedata / np.max(np.abs(wavedata))

        S = librosa.feature.melspectrogram(wavedata, sr=self.sample_rate, n_fft=self.fft_window_size,
                hop_length=self.hop_length, center=False,  # and kwargs for librosa.filters.mel
                n_mels=self.mels_number, fmin=self.min_frequency, fmax=self.max_frequency)

        return librosa.power_to_db(S, ref=np.max).T


def extract_features(data_type):
    '''
    Extracts features from audio files using log mel filters
    Parameters:
        - data_type -- (or mode) either 'train' or 'test', str
    Returns:
        - None
    '''
    # get the parameters from yaml file
    params = HParamsFromYAML('hparams.yaml', param_sets=['log_mel'])

    # get paths to the data or exit if wrong data_type is chosen
    if data_type == 'train':
        AUDIO_FILES_PATH = os.path.join(params.datasets_dir, params.train_data_dir)
        META_FILE_PATH = os.path.join(params.datasets_dir, params.metadata_dir, params.train_meta_file)
    elif data_type == 'test':
        AUDIO_FILES_PATH = os.path.join(params.datasets_dir, params.test_data_dir)
        META_FILE_PATH = os.path.join(params.datasets_dir, params.metadata_dir, params.test_meta_file)
    else:
        exit('Only \'train\' or \'test\' data types are allowed. Please use one of them. Exiting...')

    # store the feature in an h5 file
    HDF5_PATH = os.path.join(params.storage_dir, params.features_dir, 'log_mel',
            '{}.h5'.format(data_type))
    HDF5_DIRNAME = os.path.dirname(HDF5_PATH)
    if not os.path.exists(HDF5_DIRNAME):
        os.makedirs(HDF5_DIRNAME)

    # read metadata and create pandas dataframe
    metadata_df = read_csv(META_FILE_PATH)

    # initialise object with parameters for features extraction
    log_mel = LogMelSpectrogram(params)

    h5_fout = h5py.File(HDF5_PATH, 'w')

    # create a dataset for features storage
    h5_fout.create_dataset(name='feature', shape=(0, log_mel.mels_number), dtype=np.float32,
            maxshape=(None, log_mel.mels_number))

    # store indices of the beginning and the end of each feaure 
    begin_end_ind = []

    data_len = metadata_df.shape[0]
    logging.info('Extracting features from {} dataset...'.format(data_type))
    start_time = time()

    # write out the features
    for k, audio_fname in enumerate(metadata_df['fname']):
        print('Audio #{} of {}'.format(k+1, data_len), end='\r')
        path_to_audio = os.path.join(AUDIO_FILES_PATH, audio_fname)
        feature = log_mel.get_spectrogram(path_to_audio)

        begin_ind = h5_fout['feature'].shape[0]
        end_ind = begin_ind + feature.shape[0]

        h5_fout['feature'].resize((end_ind, log_mel.mels_number))
        h5_fout['feature'][begin_ind:end_ind] = feature

        begin_end_ind += [(begin_ind, end_ind)]
    logging.info('Extracting finished successfully. Elapsed time: {:.6f} s'.\
            format(time() - start_time))

    # write out the beginning/end indices
    h5_fout.create_dataset(name='begin_end_ind', dtype=np.int32, data=begin_end_ind)

    # write out audio filenames
    h5_fout.create_dataset(name='filename', dtype='S32',
            data=[s.encode() for s in metadata_df['fname'].tolist()])

    # write out labels and manually verifications if data_type is train
    if data_type == 'train':
        h5_fout.create_dataset(name='label', dtype='S32',
                data=[s.encode() for s in metadata_df['label'].tolist()])
        h5_fout.create_dataset(name='manually_verified', dtype=np.int32,
                data=metadata_df['manually_verified'].tolist())
    
    h5_fout.close()
    logging.info('The features saved to {}'.format(HDF5_PATH))
