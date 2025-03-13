import logging
import os
from datetime import datetime

import numpy as np
import torch


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt=None):
        super().__init__(fmt, datefmt)
        self.colors = {
            'DEBUG': '\033[94m',  # Blue
            'INFO': '\033[92m',   # Green
            'WARNING': '\033[93m', # Yellow
            'ERROR': '\033[91m',  # Red
            'CRITICAL': '\033[41m'  # Red background
        }

    def format(self, record: logging.LogRecord) -> str:
        color = self.colors.get(record.levelname, '\033[0m')
        # Format the prefix including date-time and log level
        prefix = f"{self.formatTime(record, self.datefmt)} {record.levelname}"
        colored_prefix = f"{color}{prefix}\033[0m"
        message = super().format(record)
        find_record = message.find(record.levelname)
        left_over = message[find_record+len(record.levelname):]
        message = colored_prefix + left_over

        # Replace the entire prefix with the colored version
        return message
        
# Set all Seeds
def set_seeds(seed: int):
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)

def create_logger(name: str) -> logging.Logger:
    # Check if .log folder exists if ot crea
    if not os.path.exists(f"logs/"):
        os.makedirs(f"logs/", exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(f"logs/{name}.log", mode="w")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    # formatter = logging.Formatter(
    #     "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # )
    formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# Creat decorator with dates and reason saying why a function is deprecated
def deprecated(reason, date):
    def decorator(func):
        def wrapper(*args, **kwargs):
            raise NotImplementedError(
                f"{func.__name__} has deprecated since {date} and will be removed in the future.\n"
                f"Reason: {reason}\n"
            )

        return wrapper

    return decorator

def inspect_array(prefix: str, arr: np.ndarray, verbose: bool = False):
    print(f"({prefix}): Shape:", arr.shape)
    print(f"({prefix}): Data type:", arr.dtype)
    print(f"({prefix}): Size:", arr.size)
    print(f"({prefix}): Number of dimensions:", arr.ndim)
    if verbose:
        print(f"({prefix}): Array:\n", arr)

def inspect_tensor(prefix: str, tensor: torch.Tensor, verbose: bool = False):
    print(f"({prefix}): Shape:", tensor.shape)
    print(f"({prefix}): Data type:", tensor.dtype)
    print(f"({prefix}): Size:", tensor.size())
    print(f"({prefix}): Number of dimensions:", tensor.ndim)
    print(f"({prefix}): Tensor:\n", tensor) 
    if verbose:
        print(f"({prefix}): Array:\n", tensor)

def array_to_csv(arr: np.ndarray, save_path: str):
    np.savetxt(save_path, arr, delimiter=",")

def calculate_correlation(vec: np.ndarray, mat:np.ndarray) -> np.ndarray:
    all_pc_corr_scores = []
    assert len(vec.shape) == 2 and vec.shape[-1] == 1, f"The vector should be a 2d array with shape (2, num_features) but is {vec.shape}"
    assert mat.shape[0] == vec.shape[0], f"The matrix should have the same number of features as the vector but is {mat.shape} and {vec.shape}"

    for i in range(mat.shape[-1]):
        print(f"Calculating correlation for column {i}")
        col_i_timeseries = mat[:,i]
        corr_i = np.corrcoef(col_i_timeseries, vec.squeeze())[0,1]
        all_pc_corr_scores.append(corr_i)
        # ensure corr_i is not nan
        # if np.isnan(corr_i):
            # print(f"Correlation is NaN for column {i}")
            # # Dump the data
            # inspect_array("mat", mat)
            # print(f"Peek into vec: {vec}")
            # inspect_array("vec", vec)
            # print(f"Peek into col_i_timeseries: {col_i_timeseries}")
        assert not np.isnan(corr_i), f"Correlation is NaN for column {i}"
    return np.abs(np.array(all_pc_corr_scores))
