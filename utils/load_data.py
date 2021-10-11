"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: A library for data loaders.
"""

import os

import datatable as dt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class Loader(object):
    """ Data loader """

    def __init__(self, config, dataset_name, drop_last=True, kwargs={}):
        """Pytorch data loader

        Args:
            config (dict): Dictionary containing options and arguments.
            dataset_name (str): Name of the dataset to load
            drop_last (bool): True in training mode, False in evaluation.
            kwargs (dict): Dictionary for additional parameters if needed

        """
        # Get batch size
        batch_size = config["batch_size"]
        # Get config
        self.config = config
        # Set the paths
        paths = config["paths"]
        # data > dataset_name
        file_path = os.path.join(paths["data"], dataset_name)
        # Get the datasets
        train_dataset, test_dataset, validation_dataset = self.get_dataset(dataset_name, file_path)
        # Set the loader for training set
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, **kwargs)
        # Set the loader for test set
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
        # Set the loader for validation set
        self.validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, **kwargs)
        

    def get_dataset(self, dataset_name, file_path):
        """Returns training, validation, and test datasets"""
        # Create dictionary for loading functions of datasets.
        # If you add a new dataset, add its corresponding dataset class here in the form 'dataset_name': ClassName
        loader_map = {'default_loader': TabularDataset}
        # Get dataset. Check if the dataset has a custom class. 
        # If not, then assume a tabular data with labels in the first column
        dataset = loader_map[dataset_name] if dataset_name in loader_map.keys() else loader_map['default_loader']
        # Training and Validation datasets
        train_dataset = dataset(self.config, datadir=file_path, dataset_name=dataset_name, mode='train')
        # Test dataset
        test_dataset = dataset(self.config, datadir=file_path, dataset_name=dataset_name, mode='test')
        # validation dataset
        validation_dataset = dataset(self.config, datadir=file_path, dataset_name=dataset_name, mode="validation")
        # Return
        return train_dataset, test_dataset, validation_dataset


class ToTensorNormalize(object):
    """Convert ndarrays to Tensors."""
    def __call__(self, sample):
        # Assumes that min-max scaling is done when pre-processing the data
        return torch.from_numpy(sample).float()


class TabularDataset(Dataset):
    def __init__(self, config, datadir, dataset_name, mode='train', transform=ToTensorNormalize()):
        """Dataset class for tabular data format.

        Args:
            config (dict): Dictionary containing options and arguments.
            datadir (str): The path to the data directory
            dataset_name (str): Name of the dataset to load
            mode (bool): Defines whether the data is for Train, Validation, or Test mode
            transform (func): Transformation function for data
            
        """

        self.config = config
        self.mode = mode
        self.paths = config["paths"]
        self.dataset_name = dataset_name
        self.data_path = os.path.join(self.paths["data"], dataset_name)
        self.data, self.labels = self._load_data()
        self.transform = transform

    def __len__(self):
        """Returns number of samples in the data"""
        return len(self.data)

    def __getitem__(self, idx):
        """Returns batch"""
        sample = self.data[idx]
        cluster = int(self.labels[idx])
        return sample, cluster

    def _load_data(self):
        """Loads one of many available datasets, and returns features and labels"""

        if self.dataset_name.lower() in ["mnist"]:
            x_train, y_train, x_test, y_test = self._load_mnist()
        else:
            print(f"Given dataset name is not found. Check for typos, or missing condition "
                  f"in _load_data() of TabularDataset class in utils/load_data.py .")
            exit()

        # Define the ratio of training-validation split, e.g. 0.8
        training_data_ratio = self.config["training_data_ratio"]
        
        # If validation is on, and trainin_data_ratio==1, stop and warn
        if self.config["validate"] and training_data_ratio >= 1.0:
            print(f"training_data_ratio must be < 1.0 if you want to run validation during training.")
            exit()            

        # Shuffle indexes of samples to randomize training-validation split
        idx = np.random.permutation(x_train.shape[0])

        # Divide training and validation data : 
        # validation data = training_data_ratio:(1-training_data_ratio)
        tr_idx = idx[:int(len(idx) * training_data_ratio)]
        val_idx = idx[int(len(idx) * training_data_ratio):]

        # Validation data
        x_val = x_train[val_idx, :]
        y_val = y_train[val_idx]
        
        # Training data
        x_train = x_train[tr_idx, :]
        y_train = y_train[tr_idx]

        # Update number of classes in the config file in case that it is not correct.
        n_classes = len(list(set(y_train.reshape(-1, ).tolist())))
        if self.config["n_classes"] != n_classes:
            self.config["n_classes"] = n_classes
            print(f"{50 * '>'} Number of classes changed "
                  f"from {self.config['n_classes']} to {n_classes} {50 * '<'}")

        # Check if the values of features are small enough to work well for neural network
        if np.max(np.abs(x_train)) > 10:
            print(f"Pre-processing of data does not seem to be correct. "
                  f"Max value found in features is {np.max(np.abs(x_train))}\n"
                  f"Please check the values of features...")
            exit()
        
        # Select features and labels, based on the mode
        if self.mode == "train":
            data = x_train
            labels = y_train
        elif self.mode == "validation":
            data = x_val
            labels = y_val
        elif self.mode == "test":
            data = x_test
            labels = y_test
        else:
            print(f"Something is wrong with the data mode. "
                  f"Use one of three options: train, validation, and test.")
            exit()
        
        # Return features, and labels
        return data, labels


    def _load_mnist(self):
        """Loads MNIST dataset"""
        
        # Overwrite the datapath since the code will be pushed with mnist 
        # dataset for demo and testing (using /results/mnist_dummy/) purposes. 
        # We don't want to have a separate mnist_dummy data folder
        self.data_path = os.path.join("./data/", "mnist")
        
        with open(self.data_path + '/train.npy', 'rb') as f:
            x_train = np.load(f)
            y_train = np.load(f)

        with open(self.data_path + '/test.npy', 'rb') as f:
            x_test = np.load(f)
            y_test = np.load(f)

        x_train = x_train.reshape(-1, 28 * 28) / 255.
        x_test = x_test.reshape(-1, 28 * 28) / 255.

        return x_train, y_train, x_test, y_test