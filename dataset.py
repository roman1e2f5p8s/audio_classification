import numpy as np
import torch
from torch.utils.data import Dataset
from hparams import HParamsFromYAML

class HyperParams:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size


class AudioDataset(Dataset):
    '''
    Loads and characterises a dataset for PyTorch
    '''
    def __init__(self, hyper_params, data_dir, file_names):
        '''
        Initialisation
        Arguments:
            - hyper_params -- object that stores hyperparameters for configuration, HParamsFromYAML
            - data_dir -- path to directory with data, str
            - file_names -- list of names of files in data_dir, list() of strings
        '''
        self.hyper_params = hyper_params
        self.data_dir = data_dir
        self.file_names = file_names
        self.file_names_indeces = np.arange(len(file_names))
    
    def __len__(self):
        '''
        Denotes the number of batches drawn in an epoch.
        If batch_size is 1, then the number of batches is equal the total number of samples.
        batch_size denotes the number of samples contained in each generated batch
        Parameters:
            - no input parameters
        Returns:
            - number of batches drawn in each epoch, int
        '''
        return int(np.ceil(len(self.file_names) / self.hyper_params.batch_size))

    def __getitem__(self, index):
        '''
        Generates one sample of data
        Parameters:
            - index -- number of batch, int
        Returns:
            - generated batch of the data: X and y,  
        '''
        indeces = self.file_names_indeces[index * self.hyper_params.batch_size : \
                (index + 1) * self.hyper_params.batch_size]
        file_names = [self.file_names[k] for k in indeces]
        return self.generate_data(file_names)
        X = self.data[index].float()
        y = self.labels[index]
        return X, y


hparams = HParamsFromYAML('./hparams.yaml', 'CNN')
ad = AudioDataset(hparams, '', [i for i in range(1600)])
print(ad.__len__())
