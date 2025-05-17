import torch
from torch.utils.data import Dataset
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class TimeCaptionDataset(Dataset):
    def __init__(self, root_path, flag='train', data_path='timecaption.json', scale=True):
        assert flag in ['train', 'test', 'val', 'all'], "flag must be one of ['train', 'test', 'val', 'all']"
        type_map = {'train': 0, 'val': 1, 'test': 2, 'all': 3}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.scalers = []  # Used to save the standardizers
        self.__read_data__()

    def __read_data__(self):
        """
        Read and process the data.
        """
        try:
            with open(os.path.join(self.root_path, self.data_path), 'r') as f:
                data_list = json.load(f)
        except FileNotFoundError:
            print(f"Error: File {os.path.join(self.root_path, self.data_path)} not found.")
            return
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {os.path.join(self.root_path, self.data_path)}.")
            return

        all_series = [item['series'] for item in data_list]
        all_annotations = [item['annotations'] for item in data_list]

        total_samples = len(all_series)
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1

        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)

        border1s = [0, train_size, train_size + val_size, 0]
        border2s = [train_size, train_size + val_size, total_samples, total_samples]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        scaled_series = []
        if self.scale:
            for series in all_series:
                scaler = StandardScaler()
                series = np.array(series).reshape(-1, 1)
                scaled = scaler.fit_transform(series).flatten()
                scaled_series.append(scaled)
                if self.set_type == 0:  # Save the standardizer only during the training phase
                    self.scalers.append(scaler)
        else:
            scaled_series = [np.array(series) for series in all_series]

        self.data_x = scaled_series[border1:border2]
        self.data_y = all_annotations[border1:border2]

    def __getitem__(self, index):
        x = torch.tensor(self.data_x[index], dtype=torch.float32)
        y = self.data_y[index]
        return x, y

    def __len__(self):
        return len(self.data_x)