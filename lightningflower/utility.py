"""LightningFlower Utility"""
import inspect
import numpy as np
import os
import secrets
from pathlib import Path


def id_generator(size=6):
    """Formatted random string id generator"""
    return ''.join(str(secrets.randbelow(10)) for _ in range(size))


def printf(message):
    """Formatted printing, e.g. [filenameOfCallingFile.py]: your text"""
    print("[" + os.path.basename(inspect.stack()[1].filename) + "]: " + message)


def load_params(full_filepath):
    params = None
    print("Loading weights from " + full_filepath)
    # load weights
    if os.path.isfile(full_filepath):
        weights = np.load(full_filepath, allow_pickle=True)
        if weights is not None:
            params = weights.f.arr_0.item(0)
        if params is None:
            printf("Failed loading weights")
        else:
            printf("Success loading weights")
    return params


def boolean_string(s):
    """String to bool conversion"""
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def create_dir(path):
    """Creating a directory, check if directory exists"""
    return Path(path).mkdir(parents=True, exist_ok=True)
