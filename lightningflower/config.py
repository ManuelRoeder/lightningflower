"""LightningFlower Default Configuration"""
from enum import Enum


class DatasetConfig(Enum):
    NON_IID = 0
    IID = 1
    SAMPLED = 2


class LightningFlowerDefaults(object):
    NR_ROUNDS = 2
    GRPC_MAX_MSG_LENGTH = 1073741824  # 1024x1024x1024
    HOST_ADDRESS = "localhost:8081"
    CLIENT_ID = 1
    WEIGHTS_FILENAME = "weights.npz"
    WEIGHTS_FOLDER = "./saved_weights"
    DATASET_FOLDER = './data'
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_TEST = 32
    DATASET_CONFIG = DatasetConfig.IID
    NUM_WORKERS = 2
    NUM_CLIENTS = 2
