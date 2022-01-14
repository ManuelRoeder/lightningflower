"""LightningFlower Data Module"""
import numpy as np
import os
import torch.utils
from torch.utils.data import DataLoader, SubsetRandomSampler
from lightningflower.config import LightningFlowerDefaults


class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms."""

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


class CustomNumpyDataset(torch.utils.data.Dataset):
    """NumpyDataset with support of transforms."""

    def __init__(self, path, train):
        self.path = path
        if train:
            self.tensors = (
                torch.from_numpy(np.load(os.path.join(path, "X_train.npy"))),
                torch.from_numpy(np.load(os.path.join(path, "y_train.npy")))
            )

        else:
            self.tensors = (
                torch.from_numpy(np.load(os.path.join(path, "X_test.npy"))),
                torch.from_numpy(np.load(os.path.join(path, "y_test.npy")))
            )
        self.data = self.tensors[0].squeeze().float()
        self.targets = self.tensors[-1].float()

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


class LightningFlowerData:
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LightningFlowerData")
        parser.add_argument("--dataset_path", type=str, default=LightningFlowerDefaults.DATASET_FOLDER)
        parser.add_argument("--batch_size_train", type=int, default=LightningFlowerDefaults.BATCH_SIZE_TRAIN)
        parser.add_argument("--batch_size_test", type=int, default=LightningFlowerDefaults.BATCH_SIZE_TEST)
        parser.add_argument("--dataset_config", type=int, default=LightningFlowerDefaults.DATASET_CONFIG,
                            help="0 for NON-IID, 1 for IID, 2 for SAMPLED")
        parser.add_argument("--num_workers", type=int, default=LightningFlowerDefaults.NUM_WORKERS)
        return parent_parser

    @staticmethod
    def split_per_client(dataset, client_id, num_clients, shuffle=False, stratify=False):
        """Splits the dataset according to the partition size calculated with client_id/num_clients"""
        if int(client_id) >= num_clients:
            return None

        # partition size based on numer of clients
        partition_size = 1.0 / num_clients
        # indices passed to subset random sampler
        train_idx = []
        # training examples
        num_samples = len(dataset)
        # list of indices
        indices = list(range(num_samples))
        # split size based on partitions
        split = int(np.floor(partition_size * num_samples))

        # stratified split implementation
        if stratify:
            # on-demand import of sklearn lib
            from sklearn.model_selection import train_test_split
            # Split the indices in a stratified way
            train_idx, _ = train_test_split(indices,
                                            train_size=partition_size,
                                            stratify=dataset.targets,
                                            shuffle=shuffle)
            if shuffle:
                print("Client " + str(client_id) + "/" + str(num_clients) +
                      " received a random shuffled stratified split of size " + str(split))
            else:
                print("Client " + str(client_id) + "/" + str(num_clients) +
                      " received a stratified split of size " + str(split))
        else:
            if shuffle:
                np.random.shuffle(indices)
                # indices are shuffled, just use the first split
                train_idx = indices[split:]
            else:
                idx_from, idx_to = client_id * split, (client_id + 1) * split - 1
                train_idx = indices[idx_from:idx_to]
            if shuffle:
                print("Client " + str(client_id) + "/" + str(
                    num_clients) + " receives a random shuffled split of size " + str(split))
            else:
                print("Client " + str(client_id) + "/" + str(num_clients) + " receives data from index " + str(
                    idx_from) + " to " + str(idx_to))
        return SubsetRandomSampler(train_idx)
