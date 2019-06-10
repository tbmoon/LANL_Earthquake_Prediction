import numpy as np
import pandas as pd
from scipy.signal import spectrogram
import torch
from torch.utils.data import Dataset, DataLoader


class QuakeDataset(Dataset):

    def __init__(self, input_dir, csv_file, n_samples, time_length, time_size):
        """
        fixed time length for sampling
        """
        self.df_quake = pd.read_csv(input_dir + '/' + csv_file)
        self.quake_length = len(self.df_quake)
        self.n_samples = n_samples
        self.time_length = time_length
        self.time_size = time_size
        self.quake_samples = np.random.randint(0, self.quake_length - time_length * time_size, n_samples)
        
        self.nperseg = 256 # default 256
        self.noverlap = self.nperseg // 4 # default: nperseg // 8
        self.fs = 4000000 # raw signal sample rate is 4MHz
        self.window = 'triang'
        self.scaling = 'density' # {'density', 'spectrum'}
        self.detrend = 'linear' # {'linear', 'constant', False}
        self.eps = 1e-11
        
    def __len__(self):
        
        return self.n_samples

    def __getitem__(self, idx):
        start_idx = self.quake_samples[idx]
        end_idx = start_idx + self.time_length * self.time_size
        
        '''
        amplitude: [time_length, time_size]
        target: [remaining_time]
        '''
        #amplitude = self.df_quake.iloc[start_idx:end_idx,0].values.reshape(-1, self.time_size)
        #target = self.df_quake.iloc[end_idx-1:end_idx,1].values
        #return {'amplitude': torch.from_numpy(amplitude).float(), 'target': torch.from_numpy(target).float()}
        
        target = self.df_quake.iloc[end_idx-1:end_idx,1].values
        
        amplitude = self.df_quake.iloc[start_idx:end_idx,0].values        
        f, t, Sxx = spectrogram(amplitude,
                                nperseg=self.nperseg,
                                noverlap=self.noverlap,
                                fs=self.fs,
                                window=self.window,
                                scaling=self.scaling,
                                detrend=self.detrend)
        Sxx = np.log(Sxx + self.eps)
        Sxx = Sxx[:-1, :]
        Sxx = Sxx.transpose(1, 0)
        
        return {'Sxx': torch.from_numpy(Sxx).float(), 'target': torch.from_numpy(target).float()}


def get_dataloader(
    input_dir,
    csv_file_train,
    csv_file_valid,
    n_samples_train,
    n_samples_valid,
    time_length,
    time_size,
    batch_size,
    num_workers):

    quake_dataset = {
        'train': QuakeDataset(
            input_dir=input_dir,
            csv_file=csv_file_train,
            n_samples=n_samples_train,
            time_length=time_length,
            time_size=time_size),
        'valid': QuakeDataset(
            input_dir=input_dir,
            csv_file=csv_file_valid,
            n_samples=n_samples_valid,
            time_length=time_length,
            time_size=time_size)} 

    dataloaders = {x: torch.utils.data.DataLoader(
        quake_dataset[x],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers) for x in ['train', 'valid']}

    data_size = {x: len(quake_dataset[x]) for x in ['train', 'valid']}

    return dataloaders, data_size
