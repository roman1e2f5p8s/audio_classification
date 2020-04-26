## 1
# Loading modules
# from __future__ import print
import numpy as np
np.random.seed(2020)

import os
import shutil

import IPython
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook
from sklearn.model_selection import StratifiedKFold

%matplotlib inline
matplotlib.style.use('ggplot')
## 1

## 2
# Reproduce the result or not
REPRODUCIBILITY = True
## 2

## 3
# loading data
train_df = pd.read_csv('../Dataset/train.csv')
test_df = pd.read_csv('../Dataset/sample_submission.csv')
## 3

## 4
print(train_df)
print(train_df.shape)
## 4

## 5
classes = train_df['label'].unique()
print('Number of training examples: {}'.format(train_df.shape[0]))
print('Number of classes: {}'.format(len(classes)))
## 5

## 6
print('Classes:')
print(classes)
## 6

## 7
# distribution of categories
category_group = train_df.groupby(['label', 'manually_verified']).count()
print(category_group)
category_group_unstack = category_group.unstack().\
        reindex(category_group.unstack().sum(axis=1).sort_values().index)
print(category_group_unstack)
plot = category_group_unstack.plot(figsize=(16, 10), kind='bar', stacked=True,
        title='Number of audio samples per category')
plot.set_xlabel('Category')
plot.set_ylabel('Number of samples')
## 7

## 8
print('Minimum number of audio samples per category: {}'.format(min(train_df['label'].value_counts())))
print('Maximum number of audio samples per category: {}'.format(max(train_df['label'].value_counts())))
## 8

## 9
# listening to an audio file
import IPython.display as ipd
AFNAME = '../Dataset/audio_train/00044347.wav'
ipd.Audio(AFNAME)
## 9

## 10
# using wave library
import wave
AFNAME = '../Dataset/audio_test/0b0427e2.wav'
wav_f = wave.open(AFNAME)
frame_rate = wav_f.getframerate()  # sampling rate
frames_numb = wav_f.getnframes()
duration_w = frames_numb / frame_rate
print('Sampling (frame) rate: {}'.format(frame_rate))
print('Total number of samples (frames): {}'.format(frames_numb))
print('Duration: {} s'.format(duration_w))
## 10

## 11
# using scipy
from scipy.io import wavfile
rate, data = wavfile.read(AFNAME)
duration_s = data.shape[0] / rate
print('Sampling (frame) rate: {}'.format(rate))
print('Total number of samples (frames): {}'.format(data.shape[0]))
print('Duration: {} s'.format(duration_s))
print('Data:')
print(data, len(data))
## 11

## 19
# load audio using torch
import torch
import torchaudio

def normalise(waveform_tensor):
    # Subtract the mean, and scale to the interval [-1,1]
    wf = waveform_tensor # - waveform_tensor.mean()
    return wf / wf.abs().max()
# si, ei = torchaudio.info(AFNAME)
# rate, channels, encoding = si.rate, si.channels, ei.encoding
# print(rate, channels, encoding)
wavedata, sampling_rate = torchaudio.load_wav(AFNAME, normalization=True)  # True is default
## 19
duration = wavedata.size()[1] / sampling_rate
print('RAW DATA')
print('Shape of wavedata (channel, time): {}'.format(wavedata.size()))
print('Sampling rate of wavedata: {}'.format(sampling_rate))
print('Duration: {} s'.format(duration))
# check whether the wavedata tensor is in the interval [-1, 1]
print('Min of wavedata: {}'.format(wavedata.min()))
print('Max of wavedata: {}'.format(wavedata.max()))
print('Mean of wavedata: {}'.format(wavedata.mean()))
print(wavedata)
plt.figure()
plt.plot(wavedata.t().numpy())
print()

# resample data
NEW_SAMPLING_RATE = 16000
resample = torchaudio.transforms.Resample(sampling_rate, NEW_SAMPLING_RATE)
wavedata = resample(wavedata[0, :].view(1, -1))
duration = wavedata.size()[1] / resample.new_freq
print('RESAMPLED DATA')
print('Shape of wavedata (channel, time): {}'.format(wavedata.size()))
print('Sampling rate of wavedata: {}'.format(resample.new_freq))
print('Duration: {} s'.format(duration))
# check whether the wavedata tensor is in the interval [-1, 1]
print('Min of wavedata: {}'.format(wavedata.min()))
print('Max of wavedata: {}'.format(wavedata.max()))
print('Mean of wavedata: {}'.format(wavedata.mean()))
print(wavedata)
plt.figure()
plt.plot(wavedata[0, :].numpy())
print()

# normalise data
wavedata = normalise(wavedata)
duration = wavedata.size()[1] / NEW_SAMPLING_RATE
print('NORMALISED DATA')
print('Shape of wavedata (channel, time): {}'.format(wavedata.size()))
print('Sampling rate of wavedata: {}'.format(NEW_SAMPLING_RATE))
print('Duration: {} s'.format(duration))
# check whether the wavedata tensor is in the interval [-1, 1]
print('Min of wavedata: {}'.format(wavedata.min()))
print('Max of wavedata: {}'.format(wavedata.max()))
print('Mean of wavedata: {}'.format(wavedata.mean()))
print(wavedata)
plt.figure()
plt.plot(wavedata.t().numpy())
print()

# '''
def get_audio_data(filename):
    wavedata, sampling_rate = torchaudio.load_wav('../Dataset/audio_train/'+ filename)
    # wavedata = normalise(wavedata)
    # print('Shape of wavedata(channel, time): {}'.format(wavedata.size()))
    # print('Sampling rate of wavedata: {}'.format(sampling_rate))
    duration = wavedata.size()[1] / sampling_rate
    # print('Duration: {} s'.format(duration))
    # check whether the wavedata tensor is in the interval [-1, 1]
    # print('Min of wavedata: {}'.format(wavedata.min()))
    # print('Max of wavedata: {}'.format(wavedata.max()))
    # print('Mean of wavedata: {}'.format(wavedata.mean()))
    # print(wavedata)
    num_channels, num_frames = wavedata.size()
    # print(num_channels, num_frames, sampling_rate, duration, wavedata)
    # plt.figure()
    # plt.plot(wavedata.t().numpy())
    # print(type(wavedata.t().numpy()))
    return num_channels, num_frames, sampling_rate, duration  # , wavedata.t().numpy()
train_df['nchannels'], train_df['nframes'], train_df['srate'], train_df['dur'] =\
        zip(*train_df['fname'].map(get_audio_data))
print(train_df)
# '''
## 19

##
nchannels = train_df['nchannels'].unique()
print('Channels: {}'.format(nchannels))
sampling_rates = train_df['srate'].unique()
print('Sampling rates: {}'.format(sampling_rates))
##

## 12
plt.plot(data, '_')
## 12

## 13
plt.figure(figsize=(10, 4))
plt.plot(data[:500], '.')
plt.plot(data[:500], '-')
## 13

## 15
train_df['nframes'] = train_df['fname'].apply(lambda f: wave.open('../Data/audio_train/' + f).getnframes())
train_df['nf'] = train_df['fname'].apply(lambda f: f)
test_df['nframes'] = test_df['fname'].apply(lambda f: wave.open('../Data/audio_test/' + f).getnframes())
print(train_df)
print(test_df['nframes'])
## 15

## 16
_, ax = plt.subplots(figsize=(16, 4))
sns.violinplot(ax=ax, data=train_df, x='label', y='nframes')
plt.xticks(rotation=90)
plt.title('Distribution of audio frames per label', fontsize=16)
plt.show()
## 16

## 17
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
train_df.nframes.hist(bins=100, ax=axes[0])
test_df.nframes.hist(bins=100, ax=axes[1])
plt.suptitle('Frame length distribution in train_df and test data', ha='center', fontsize='large')
## 17

## 18
abnormal_len = [707364, 353682, 138474, 184338]
for l in abnormal_len:
    abnormal_afname = test_df.loc[test_df.nframes == l, 'fname'].values
    print('Frame length: {}. Number of files: {}'.format(l, abnormal_afname.shape[0]))
    afname = np.random.choice(abnormal_afname)
    print('Playing {}'.format(afname))
    ipd.Audio('../Data/audio_test/' + afname)
## 18

## 19
# load audio using torch
import torchaudio
waveform, sampling_rate = torchaudio.load(AFNAME)
print('Shape of waveform (channel, time): {}'.format(waveform.size()))
print('Sampling rate of waveform: {}'.format(sampling_rate))
print('Duration: {} s'.format(waveform.size()[1] / sampling_rate))
plt.figure()
plt.plot(waveform.t().numpy())
## 19

## 20
# resample audio using torch
NEW_SAMPLING_RATE = 16000
waveform_transformed = torchaudio.transforms.Resample(sampling_rate, NEW_SAMPLING_RATE)\
    (waveform[0, :].view(1, -1))
print('Shape of transformed waveform (channel, time): {}'.format(waveform_transformed.size()))
print('Duration: {} s'.format(waveform_transformed.size()[1] / NEW_SAMPLING_RATE))
plt.figure()
plt.plot(waveform_transformed[0, :].numpy())
## 20

## 21
# check whether the waveform tensor is in the interval [-1, 1]
print('Min  of waveform: {}'.format(waveform_transformed.min()))
print('Max  of waveform: {}'.format(waveform_transformed.max()))
print('Mean of waveform: {}'.format(waveform_transformed.mean()))
## 21

## 22
def normalise(waveform_tensor):
    # Subtract the mean, and scale to the interval [-1,1]
    wf = waveform_tensor - waveform_tensor.mean()
    return wf / wf.abs().max()
# waveform_transformed = normalise(waveform_transformed)
# print('Min    of waveform: {}'.format(waveform_transformed.min()))
# print('Max    of waveform: {}'.format(waveform_transformed.max()))
# print('Mean of waveform: {}'.format(waveform_transformed.mean()))
## 22

## 23
# load audio using librosa
import librosa
import librosa.display
print(librosa.__version__)
AFNAME = '../Dataset/audio_test/0b0427e2.wav'
waveform, sampling_rate = librosa.core.load(AFNAME, sr=16000, res_type='kaiser_fast')
print('Shape of waveform (channel, time): {}'.format(len(waveform)))
print('Sampling rate of waveform: {}'.format(sampling_rate))
print('Duration: {} s'.format(librosa.get_duration(y=waveform)))
plt.figure()
librosa.display.waveplot(waveform)
## 23
