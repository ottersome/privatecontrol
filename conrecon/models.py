import logging
import os

import torch
import torch.nn.functional as F
from torch import nn


class RecoveryTrans(nn.Module):
    def __init__(self, input_size, state_size, num_outputs, time_steps):
        super().__init__()

    def forward(self, x):
        pass


class SimpleModel(nn.Module):
    def __init__(self, input_size, state_size, num_outputs, time_steps):
        super().__init__()
        # Just do two layers of linear and relu
        self.layer1 = nn.Linear(input_size, state_size)
        self.layer2 = nn.Linear(state_size, num_outputs)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class TModel(nn.Module):
    def __init__(
        self,
        d_model,
        output_size,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
        memory_casual=False,
    ) -> None:
        super().__init__()
        self.memory_casual = memory_casual
        self.decoder_projection = nn.Linear(1, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.final_layer = nn.Linear(d_model, output_size)

    def forward(self, src, tgt):
        transd_tgt = self.decoder_projection(tgt)
        transy = self.transformer(src, transd_tgt)
        return self.final_layer(transy)

        # return self.transformer(src, tgt, memory_casual=self.memory_casual)


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