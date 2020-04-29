import os
import h5py
from time import time
import numpy as np
from pandas import read_csv
import librosa
import logging

from .hparams import HParamsFromYAML


class Spectrogram():
    '''
    Stores the common parameters for log-mel, mfcc, or chroma spectrograms.
    '''
    def __init__(self, params):
        '''
        Initialisation
        Arguments:
            - params -- parameters, hparams.HParamsFromYAML
        Attributes:
            - sample_rate -- target sampling rate for feature extraction, int > 0
            - fft_window_size -- length of the FFT window, int > 0
            - hop_length -- number of samples between successive frames, int > 0
            - min_frequency -- lowest frequency (in Hz), float >= 0
            - max_frequency -- highest frequency (in Hz), float >= 0. If None, use sample_rate // 2
        '''
        self.sample_rate = params.sample_rate
        self.fft_window_size = params.fft_window_size
        self.hop_length = params.hop_length
        self.min_frequency = params.min_frequency
        self.max_frequency = params.sample_rate // 2 if params.max_frequency is None \
                else params.max_frequency

    def get_wavedata(self, path_to_audio):
        '''
        Reads and resamples audio using the librosa library, normalises the audiodata
        Parameters:
            - path_to_audio -- path to audio file, str
        Returns:
            - wavedata -- audio time series, numpy.ndarray
        '''
        wavedata, _ = librosa.core.load(path_to_audio, sr=self.sample_rate)
        return wavedata / np.max(np.abs(wavedata))


class LogMelSpectrogram(Spectrogram):
    '''
    Class of log10 mel-scaled spectrogram
    '''
    def __init__(self, params):
        '''
        Initialisation
        Arguments:
            - params -- parameters, hparams.HParamsFromYAML
        Additional attributes:
            - mels_number -- number of Mel bands to generate, int > 0
        '''
        super(LogMelSpectrogram, self).__init__(params)
        self.mels_number = params.mels_number
        
    def get_spectrogram(self, path_to_audio, transpose=True):
        '''
        Reads and resamples audio using the librosa library, normalises the audiodata
        and returns log mel spectrogram using parameters obtained during the initialisation
        Parameters:
            - path_to_audio -- path to audio file, str
            - transpose -- transpose the ouput array or not, bool. Defaults to True
        Returns:
            - log10 mel spectrogram, numpy.ndarray of shape
                (len(wavedata)//self.hop_length-1, self.mels_number) if not transposed
        '''
        wavedata = self.get_wavedata(path_to_audio)

        S = librosa.feature.melspectrogram(
                y=wavedata,
                sr=self.sample_rate,
                n_fft=self.fft_window_size,
                n_mels=self.mels_number,
                hop_length=self.hop_length,
                center=False,  # and kwargs for librosa.filters.mel
                fmin=self.min_frequency,
                fmax=self.max_frequency
                )
        logS = librosa.power_to_db(S, ref=np.max)

        return logS.T if transpose else logS


class MfccSpectrogram(Spectrogram):
    '''
    Class of mfcc spectrogram.
    '''
    def __init__(self, params):
        '''
        Initialisation
        Arguments:
            - params -- parameters, hparams.HParamsFromYAML
        Additional attributes:
            - mfcc_number -- number of MFCCs to return, int > 0
        '''
        super(MfccSpectrogram, self).__init__(params)
        self.mfcc_number = params.mfcc_number
        
    def get_spectrogram(self, path_to_audio, transpose=True):
        '''
        Reads and resamples audio using the librosa library, normalises the audiodata
        and returns mfcc spectrogram using parameters obtained during the initialisation
        Parameters:
            - path_to_audio -- path to audio file, str
            - transpose -- transpose the ouput array or not, bool. Defaults to True
        Returns:
            - mfcc spectrogram, numpy.ndarray of shape
                (self.mfcc_number, len(wavedata)//self.hop_length-1) if not transposed
        '''
        wavedata = self.get_wavedata(path_to_audio)

        S = librosa.feature.mfcc(
                y=wavedata,
                sr=self.sample_rate,
                n_fft=self.fft_window_size,
                n_mfcc=self.mfcc_number,
                hop_length=self.hop_length,
                center=False,  # and kwargs for librosa.filters.mel
                fmin=self.min_frequency,
                fmax=self.max_frequency
                )

        return S.T if transpose else S


class ChromaSpectrogram(Spectrogram):
    '''
    Class of chroma_stft spectrogram.
    '''
    def __init__(self, params):
        '''
        Initialisation
        Arguments:
            - params -- parameters, hparams.HParamsFromYAML
        Additional attributes:
            - chroma_number -- number of chroma bins to produce, int > 0
        '''
        super(ChromaSpectrogram, self).__init__(params)
        delattr(self, 'min_frequency')
        delattr(self, 'max_frequency')
        self.chroma_number = params.chroma_number
        
    def get_spectrogram(self, path_to_audio, transpose=True):
        '''
        Reads and resamples audio using the librosa library, normalises the audiodata
        and returns chroma_stft spectrogram using parameters obtained during the initialisation
        Parameters:
            - path_to_audio -- path to audio file, str
            - transpose -- transpose the ouput array or not, bool. Defaults to True
        Returns:
            - chroma_stft spectrogram, numpy.ndarray of shape
                (self.chroma_number, len(wavedata)//self.hop_length-1) if not transposed
        '''
        wavedata = self.get_wavedata(path_to_audio)

        S = librosa.feature.chroma_stft(
                y=wavedata,
                sr=self.sample_rate,
                n_fft=self.fft_window_size,
                n_chroma=self.chroma_number,
                hop_length=self.hop_length,
                center=False,
                )

        return S.T if transpose else S


def extract_features(data_type, features_type):
    '''
    Extracts features from audio files using log-mel or mfcc
    Parameters:
        - data_type -- (or mode) either 'train' or 'test', str
        - features_type -- either 'log_mel' or 'mfcc', str
    Returns:
        - None
    '''
    if not features_type in ['log_mel', 'mfcc', 'chroma']:
        print('Features type \"{}\" is not supported.\n'.format(features_type) + \
                'Use either \"log_mel\", \"mfcc\", or \"chroma\" features type. Exiting.')
    # get the parameters from yaml file
    params = HParamsFromYAML('hparams.yaml', param_sets=[features_type])

    # get paths to the data or exit if wrong data_type is chosen
    if data_type == 'train':
        AUDIO_FILES_PATH = os.path.join(params.datasets_dir, params.train_data_dir)
        META_FILE_PATH = os.path.join(params.datasets_dir, params.metadata_dir, params.train_meta_file)
    elif data_type == 'test':
        AUDIO_FILES_PATH = os.path.join(params.datasets_dir, params.test_data_dir)
        META_FILE_PATH = os.path.join(params.datasets_dir, params.metadata_dir, params.test_meta_file)
    else:
        exit('Only \"train\" or \"test\" data types are allowed. Please use one of them. Exiting.')

    # store the feature in an h5 file
    HDF5_PATH = os.path.join(params.storage_dir, params.features_dir, features_type,
            '{}.h5'.format(data_type))
    HDF5_DIRNAME = os.path.dirname(HDF5_PATH)
    if not os.path.exists(HDF5_DIRNAME):
        os.makedirs(HDF5_DIRNAME)

    # read metadata and create pandas dataframe
    metadata_df = read_csv(META_FILE_PATH)

    # initialise object with parameters for features extraction
    if features_type == 'log_mel':
        spec = LogMelSpectrogram(params)
        spec_param = spec.mels_number
    elif features_type == 'mfcc':
        spec = MfccSpectrogram(params)
        spec_param = spec.mfcc_number
    elif features_type == 'chroma':
        spec = ChromaSpectrogram(params)
        spec_param = spec.chroma_number

    h5_fout = h5py.File(HDF5_PATH, 'w')

    # create a dataset for features storage
    h5_fout.create_dataset(name='feature', shape=(0, spec_param), dtype=np.float32,
            maxshape=(None, spec_param))

    # store indices of the beginning and the end of each feaure 
    begin_end_ind = []

    data_len = metadata_df.shape[0]
    logging.info('Extracting {} features from {} dataset...'.format(features_type, data_type))
    start_time = time()

    # write out the features
    for k, filename in enumerate(metadata_df['fname']):
        print('Audio #{} of {}'.format(k + 1, data_len), end='\r')
        path_to_audio = os.path.join(AUDIO_FILES_PATH, filename)
        feature = spec.get_spectrogram(path_to_audio)

        begin_ind = h5_fout['feature'].shape[0]
        end_ind = begin_ind + feature.shape[0]

        h5_fout['feature'].resize((end_ind, spec_param))
        h5_fout['feature'][begin_ind:end_ind] = feature

        begin_end_ind += [(begin_ind, end_ind)]
    logging.info('Extracting finished successfully. Elapsed time: {:.6f} s'.\
            format(time() - start_time))

    # write out the beginning/end indices
    h5_fout.create_dataset(name='begin_end_ind', dtype=np.int32, data=begin_end_ind)

    # write out audio filenames
    h5_fout.create_dataset(name='filename', dtype='S32',
            data=[s.encode() for s in metadata_df['fname'].tolist()])

    h5_fout.create_dataset(name='label', dtype='S32',
            data=[s.encode() for s in metadata_df['label'].tolist()])
    # write out manually verifications if data_type is train
    if data_type == 'train':
        h5_fout.create_dataset(name='manually_verified', dtype=np.int32,
                data=metadata_df['manually_verified'].tolist())
    
    h5_fout.close()
    logging.info('The features saved to {}'.format(HDF5_PATH))
