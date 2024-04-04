import os
import numpy as np
import pandas as pd
from src.load import load_multiple_trajectoryCollection_parallel_pickle as lmtp
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler



class AISDataset(Dataset):
    """
    Dataset class for loading AIS data.

    Args:
        files_list (list): List of file names.
        files_dir (str): Directory containing the files.
        transform (callable, optional): Optional transform to be applied to each sample.
        num_workers (int, optional): Number of workers for parallel processing.

    Raises:
        ValueError: If any file in the files_list is not found or if files_list is empty.

    Attributes:
        files_list (list): List of existing file paths.
        num_workers (int): Number of workers for parallel processing.
        files_dir (str): Directory containing the files.
        transform (callable): Transform to be applied to each sample.
        data (DataFrame): Concatenated data from all files.

    """

    def __init__(self, files_list, files_dir, transform=None, num_workers=16):
        #> Join the files_list with the files_dir and check for existence
        if files_list:
            absolute_paths = [os.path.join(files_dir, file) for file in files_list]
            existing_files = [file for file in absolute_paths if os.path.exists(file)]
            non_existing_files = [file for file in absolute_paths if not os.path.exists(file)]
            if non_existing_files or len(existing_files) != len(files_list): # If any file not found or the number of files in the list is not equal to the number of existing files, raise exception
                raise ValueError("Invalid files list. The following files are not found:\n", non_existing_files)
            else: # All good :)
                self.files_list = existing_files # With absolute paths
        else:
            raise ValueError("files_list is empty")
        
        self.num_workers = num_workers
        self.files_dir = files_dir
        self.transform = transform
        
        self.data = pd.DataFrame()
        #> Load the data
        dfs = lmtp(self.files_list, num_cores=self.num_workers)
        for df in dfs:
            self.data = pd.concat([self.data, df], axis=0, ignore_index=True)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.data.iloc[idx].values)
        if self.transform:
            sample = self.transform(sample)
        return sample
    

class DenoiseAutoencoderDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
        self.feature_columns = dataframe.columns.drop(['epoch', 'datetime', 'obj_id', 'traj_id', 'abs_ccs', 'stopped', 'curv'])
        
        features_dataframe = self.dataframe[self.feature_columns]

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features_dataframe.values)
        self.features = scaled_features

        # self.features = features_dataframe.values
        self.labels = self.features.copy()

        self.n_features = features_dataframe.shape[1]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx].astype(np.float32), dtype=torch.float)
        labels = torch.tensor(self.labels[idx].astype(np.float32), dtype=torch.float) 
        return features, labels
    

class DenoiseAutoencoderSequencedDataset(Dataset):
    def __init__(self, dataframe, drop_features_list, seq_len=50):
        self.seq_len = seq_len
        self.dataframe = dataframe
        self.drop_features_list = drop_features_list
        
        # Remove columns not needed for features
        self.feature_columns = dataframe.columns.drop(self.drop_features_list)
        features_dataframe = self.dataframe[self.feature_columns]

        # Normalize features
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features_dataframe.values)

        # Reshape data into (sequence_length, feature_dimension), might need padding or truncating
        self.features, self.labels = self._reshape_and_pad_sequences(scaled_features, seq_len)

        self.n_features = self.features.shape[-1]  # feature_dimension
        self.l_dataset = len(scaled_features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, labels

    def _reshape_and_pad_sequences(self, features, sequence_length):
        # Calculate needed padding
        self.total_sequences = int(np.ceil(len(features) / sequence_length))
        self.padding_needed = self.total_sequences * sequence_length - len(features)
        
        # Pad feature array
        padded_features = np.vstack([features, np.zeros((self.padding_needed, features.shape[1]))])
        
        # Reshape into (total_sequences, sequence_length, feature_dimension)
        reshaped_features = padded_features.reshape(self.total_sequences, sequence_length, -1)
        
        return reshaped_features, reshaped_features  # In a denoising autoencoder, features are the labels
    


def sliding_window_sequences(data, seq_len):
    sequences = []
    n = len(data)

    # Generate sequences for each group
    for start in range(0, n, seq_len):
        end = start + seq_len
        if end <= n:
            # If the remaining sequence is long enough, add it directly
            sequences.append(data[start:end])
        else:
            # If the remaining sequence is shorter than seq_len, pad it
            padding = np.zeros((seq_len - (n - start), data.shape[1]))
            sequence = np.vstack((data[start:n], padding))
            sequences.append(sequence)

    return sequences

class TrajectoryDataset(Dataset):
    def __init__(self, dataframe, drop_features_list, seq_len):
        self.seq_len = seq_len
        self.dataframe = dataframe.copy()
        self.drop_features_list = drop_features_list

        # Normalize features
        self.normalize_features()

        # Define feature columns and number of features here
        self.feature_columns = self.dataframe.columns.drop(self.drop_features_list)
        self.n_features = len(self.feature_columns)

        # Prepare sequences
        self.features, self.labels = self.prepare_sequences()

    def normalize_features(self):
        scaler = MinMaxScaler()
        feature_columns = self.dataframe.columns.drop(self.drop_features_list)
        self.dataframe[feature_columns] = scaler.fit_transform(self.dataframe[feature_columns])

    def prepare_sequences(self):
        processed_features = []
        feature_columns = self.dataframe.columns.drop(self.drop_features_list)

        # Group by 'obj_id' and 'traj_id' and generate sequences
        for _, group in self.dataframe.groupby(['obj_id', 'traj_id']):
            sequences = sliding_window_sequences(group[feature_columns].values, self.seq_len)
            processed_features.extend(sequences)

        return np.array(processed_features, dtype=np.float32), np.array(processed_features, dtype=np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.float)




def sliding_window_sequences_mask(data, seq_len):
    sequences = []
    masks = []  
    n = len(data)

    for start in range(0, n, seq_len):
        end = start + seq_len
        if end <= n:
            sequences.append(data[start:end])
            masks.append(np.ones((seq_len,)))  
        else:
            valid_length = n - start
            sequence = np.vstack((data[start:n], np.zeros((seq_len - valid_length, data.shape[1]))))
            sequences.append(sequence)
            mask = np.hstack((np.ones((valid_length,)), np.zeros((seq_len - valid_length,))))  # 部分1部分0的掩码表示补零情况
            masks.append(mask)

    return sequences, masks

class TrajectoryDataset_mask(Dataset):
    def __init__(self, dataframe, drop_features_list, seq_len):
        self.seq_len = seq_len
        self.dataframe = dataframe.copy()
        self.drop_features_list = drop_features_list
        self.masks = None

        # Normalize features
        self.normalize_features()

        # Define feature columns and number of features here
        self.feature_columns = self.dataframe.columns.drop(self.drop_features_list)
        self.n_features = len(self.feature_columns)

        # Prepare sequences
        self.features, self.masks = self.prepare_sequences()

    def normalize_features(self):
        scaler = MinMaxScaler()
        feature_columns = self.dataframe.columns.drop(self.drop_features_list)
        self.dataframe[feature_columns] = scaler.fit_transform(self.dataframe[feature_columns])

    def prepare_sequences(self):
        processed_features = []
        masks = []
        feature_columns = self.dataframe.columns.drop(self.drop_features_list)

        # Group by 'obj_id' and 'traj_id' and generate sequences
        for _, group in self.dataframe.groupby(['obj_id', 'traj_id']):
            sequences, group_masks  = sliding_window_sequences_mask(group[feature_columns].values, self.seq_len)
            processed_features.extend(sequences)
            masks.extend(group_masks)

        return np.array(processed_features, dtype=np.float32), np.array(masks, dtype=np.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float), torch.tensor(self.masks[idx], dtype=torch.float)
    
