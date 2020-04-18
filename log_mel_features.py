import os
import time
import h5py
import numpy as np
import pandas as pd
import librosa
# TODO: remove it later
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal

from hparams import HParamsFromYAML


# TODO: remove it later
def read_and_resample_audio(path, target_sample_rate=22050):
    '''
    Reads and resamples audio using the librosa library.
    Note, we know that all audios have 1 channels,
    so no need to do wavedata = np.mean(wavedata, axis=1)
    We could use wave, scipy.io.wavfile, or torchaudio to read audio.
    Lisbosa's method is almost as fast as scipy's, but proposes a very easy method for resampling
    during loading datafile. In total, this choice will speed up the code.
    Parameters:
        - path -- path to audio file, str
        - target_sample_rate -- target sampling rate, int > 0. Default is 22050
    Returns:
        - tuple (wavedata, sample_rate):
            - wavedata -- audiodata, numpy.ndarray of shape (number_of_frames,)
            - sample_rate -- new sampling rate (equal to target_sample_rate), int > 0
    '''
    wavedata, sample_rate = librosa.core.load(path, sr=target_sample_rate)
    return wavedata, sample_rate


# TODO: remove it later
class LogMelExtractor():
    def __init__(self, sample_rate, fft_number, mels_number, overlap,
            min_frequency=50.0, max_frequency=None):
        self.sample_rate = sample_rate
        self.fft_number = fft_number
        self.mels_number = mels_number
        self.overlap = overlap
        self.min_frequency = min_frequency
        self.max_frequency = sample_rate // 2 if max_frequency is None else max_frequency
        self.window = np.hamming(fft_number)  # use the Hamming window as a desired window
        
        self.mel_matrix = librosa.filters.mel(sr=sample_rate, n_fft=fft_number, n_mels=mels_number, 
                fmin=min_frequency, fmax=max_frequency).T
    
    def transform(self, audio):
        [f, t, x] = signal.spectral.spectrogram(audio, window=self.window, nperseg=self.fft_number, 
                noverlap=self.overlap, detrend=False, return_onesided=True, mode='magnitude') 
        x = x.T
        x = np.dot(x, self.mel_matrix)
        x = np.log(x + 1e-8)
        x = x.astype(np.float32)
        return x


class LogMelSpectrogram:
    '''
    Stores the common parameters and computes a log10 mel-scaled spectrogram.
    '''
    def __init__(self, sample_rate, fft_window_size, mels_number, hop_length,
            min_frequency=50.0, max_frequency=None):
        '''
        Initialisation
        Arguments:
            - sample_rate -- target sampling rate for feature extraction, int > 0
            - fft_window_size -- length of the FFT window, int > 0
            - mels_number -- number of Mel bands to generate, int > 0
            - hop_length -- number of samples between successive frames, int > 0
            - min_frequency -- lowest frequency (in Hz), float >= 0. Default is 50
            - max_frequency -- highest frequency (in Hz), float >= 0. Default is None. If None,
                use sample_rate // 2
        '''
        self.sample_rate = sample_rate
        self.fft_window_size = fft_window_size
        self.mels_number = mels_number
        self.hop_length = hop_length
        self.min_frequency = min_frequency
        self.max_frequency = sample_rate // 2 if max_frequency is None else max_frequency
        
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
        - data_type -- either 'train' or 'test', str
    Returns:
        - 
    '''
    # get some parameters from yaml file
    params = HParamsFromYAML('hparams.yaml', param_set='default')

    # get paths to the data or exit if wrong data_type is chosen
    if data_type == 'train':
        AUDIO_FILES_PATH = os.path.join(params.datasets_dir, params.train_data_dir)
        META_FILE_PATH = os.path.join(params.datasets_dir, params.metadata_dir, params.train_meta_file)
    elif data_type == 'test':
        AUDIO_FILES_PATH = os.path.join(params.datasets_dir, params.test_data_dir)
        META_FILE_PATH = os.path.join(params.datasets_dir, params.metadata_dir, params.test_meta_file)
    else:
        exit('Only \'train\' or \'test\' data types are allowed. Please use one of them. Exiting...')

    # we will store the feature in an h5 file
    HDF5_PATH = os.path.join(params.storage_dir, 'features', 'log_mel', '{}.h5'.format(data_type))
    HDF5_DIRNAME = os.path.dirname(HDF5_PATH)
    if not os.path.exists(HDF5_DIRNAME):
        os.makedirs(HDF5_DIRNAME)

    # read metadata and create pandas dataframe
    metadata_df = pd.DataFrame(pd.read_csv(META_FILE_PATH))
    # print(metadata_df)

    # initialise object with parameters for features extraction
    log_mel = LogMelSpectrogram(params.sample_rate, params.fft_window_size, params.mels_number,
        params.hop_length)

    h5_fout = h5py.File(HDF5_PATH, 'w')

    # create a dataset for features storage
    h5_fout.create_dataset(name='feature', shape=(0, params.mels_number), dtype=np.float32,
            maxshape=(None, params.mels_number))

    # store indices of the beginning and the end of each feaure 
    begin_end_ind = []

    data_len = metadata_df.shape[0]
    print('Extracting features from {} dataset...'.format(data_type))
    start_time = time.time()

    # write out the features
    for k, audio_fname in enumerate(metadata_df['fname']):
        print('Audio #{} of {}'.format(k+1, data_len), end='\r')
        path_to_audio = os.path.join(AUDIO_FILES_PATH, audio_fname)
        feature = log_mel.get_spectrogram(path_to_audio)

        begin_ind = h5_fout['feature'].shape[0]
        end_ind = begin_ind + feature.shape[0]

        h5_fout['feature'].resize((end_ind, params.mels_number))
        h5_fout['feature'][begin_ind:end_ind] = feature

        begin_end_ind += [(begin_ind, end_ind)]
    print('\nExtracting finished. Elapsed time: {} s'.format(time.time() - start_time))

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
    return None


f = extract_features('test')

'''
# TODO: remove it later
# add this to the notebook
AFNAME = 'Datasets/FSDKaggle2018.audio_train/75923738.wav'
params = HParamsFromYAML('hparams.yaml', param_set='default')

st = time.time()
mel = LogMelExtractor(params.sample_rate, params.fft_window_size, params.mels_number,
        params.hop_length)
(audio, _) = read_and_resample_audio(AFNAME, target_sample_rate=params.sample_rate)
audio = audio / np.max(np.abs(audio))
S1 = mel.transform(audio)
print('scipy:', time.time() - st)

st = time.time()
mel = LogMelSpectrogram(params.sample_rate, params.fft_window_size, params.mels_number,
        params.hop_length)
S2 = mel.get_spectrogram(AFNAME)
print('librosa:', time.time() - st)

fig = plt.figure(figsize=(20, 10))
gs = GridSpec(1, 2)
plt.subplot(gs[0])
plt.title('repo')
librosa.display.specshow(S1, sr=params.sample_rate, hop_length=params.hop_length,
        x_axis='time', y_axis='mel', fmin=mel.min_frequency, fmax=mel.max_frequency)
plt.colorbar()
plt.subplot(gs[1])
plt.title('my')
librosa.display.specshow(S2, sr=params.sample_rate, hop_length=params.hop_length,
        x_axis='time', y_axis='mel', fmin=mel.min_frequency, fmax=mel.max_frequency)
plt.colorbar()
# plt.show()
print(S1.shape, S2.shape)
'''
