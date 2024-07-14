import logging
import os

import torch
import torch.nn.functional as F
from torch import nn


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
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class RecoveryTrans(nn.Module):
    def __init__(self, input_size, state_size, num_outputs, time_steps):
        super().__init__()

    def forward(self, x):
        pass


class RecoveryNet(nn.Module):

    def __init__(self, input_size, state_size, num_outputs, time_steps):
        super().__init__()
        self.mean = torch.zeros(input_size, time_steps)
        self.variance = torch.zeros(input_size, time_steps)
        self.rnn = torch.nn.GRU(input_size, state_size, batch_first=True)
        # Final output layer
        self.output_layer = torch.nn.Linear(state_size, num_outputs)
        self.count = 0
        self.batch_norm = torch.nn.BatchNorm1d(num_features=input_size)
        self.logger = create_logger("RecoveryNet")

    def forward(self, x):
        # Normalize x
        # self.update(x)
        # normed_x = self.batch_norm(x)
        # norm_x = (x - self.mean) / (self.variance + 1e-8).sqrt()
        transposed_x = x.transpose(1, 2)
        self.logger.debug(f"Tranposed x looks like {transposed_x}")
        rnnout, hidden = self.rnn(transposed_x)
        self.logger.debug(f"RNN output looks like: {rnnout}")
        return self.output_layer(F.relu(rnnout)), hidden

    def update(self, x):
        self.count += 1
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        if self.count == 1:
            self.mean = batch_mean
        else:
            old_mean = self.mean
            self.mean = (old_mean * (self.count - 1) + batch_mean) / self.count
            delta = batch_mean - old_mean
            self.variance = (self.variance * (self.count - 1) + batch_var) / self.count

            # self.variance = (
            #     self.variance * (self.count - 1)
            #     + (x - old_mean - delta).pow(2).sum(dim=0)
            # ) / self.count
