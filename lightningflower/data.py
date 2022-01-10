"""LightningFlower Data Module"""

import os
import torch.utils
import numpy as np
from torch.utils.data import DataLoader
from lightningflower.config import LightningFlowerDefaults, DatasetConfig


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
    def get_loader(dataset_train, dataset_test, args):
        train_loader = None
        test_loader = None
        if args.dataset_config is DatasetConfig.NON_IID:
            # Todo Implement
            pass
        elif args.dataset_config is DatasetConfig.IID:
            train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                       batch_size=args.batch_size_train,
                                                       num_workers=args.num_workers,
                                                       drop_last=True)
            test_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                      batch_size=args.batch_size_test,
                                                      num_workers=args.num_workers,
                                                      drop_last=True)
        elif args.dataset_config is DatasetConfig.SAMPLED:
            # Todo Implement
            pass
        return train_loader, test_loader

    @staticmethod
    def split_datasets(training_dataset, num_clients, num_shards, iid, transform=None):
        """Split the whole dataset in IID or non-IID manner for distributing to clients."""

        # unsqueeze channel dimension for grayscale image datasets
        if training_dataset.data.ndim == 3:  # convert to NxHxW -> NxHxWx1
            training_dataset.data.unsqueeze_(3)
        num_categories = np.unique(training_dataset.targets).shape[0]

        if "ndarray" not in str(type(training_dataset.data)):
            training_dataset.data = np.asarray(training_dataset.data)
        if "list" not in str(type(training_dataset.targets)):
            training_dataset.targets = training_dataset.targets.tolist()

        # split dataset according to iid flag
        if iid:
            # shuffle data
            shuffled_indices = torch.randperm(len(training_dataset))
            training_inputs = training_dataset.data[shuffled_indices]
            training_labels = torch.Tensor(training_dataset.targets)[shuffled_indices]

            # partition data into num_clients
            split_size = len(training_dataset) // num_clients
            split_datasets = list(
                zip(
                    torch.split(torch.Tensor(training_inputs), split_size),
                    torch.split(torch.Tensor(training_labels), split_size)
                )
            )

            # finalize bunches of local datasets
            local_datasets = [
                CustomTensorDataset(local_dataset, transform=transform)
                for local_dataset in split_datasets
            ]
        else:
            # sort data by labels
            sorted_indices = torch.argsort(torch.Tensor(training_dataset.targets))
            training_inputs = training_dataset.data[sorted_indices]
            training_labels = torch.Tensor(training_dataset.targets)[sorted_indices]

            # partition data into shards first
            shard_size = len(training_dataset) // num_shards  # 300
            shard_inputs = list(torch.split(torch.Tensor(training_inputs), shard_size))
            shard_labels = list(torch.split(torch.Tensor(training_labels), shard_size))

            # sort the list to conveniently assign samples to each clients from at least two classes
            shard_inputs_sorted, shard_labels_sorted = [], []
            for i in range(num_shards // num_categories):
                for j in range(0, ((num_shards // num_categories) * num_categories),
                               (num_shards // num_categories)):
                    shard_inputs_sorted.append(shard_inputs[i + j])
                    shard_labels_sorted.append(shard_labels[i + j])

            # finalize local datasets by assigning shards to each client
            shards_per_clients = num_shards // num_clients
            local_datasets = [
                CustomTensorDataset(
                    (
                        torch.cat(shard_inputs_sorted[i:i + shards_per_clients]),
                        torch.cat(shard_labels_sorted[i:i + shards_per_clients]).long()
                    ),
                    transform=transform
                )
                for i in range(0, len(shard_inputs_sorted), shards_per_clients)
                ]
        return local_datasets
