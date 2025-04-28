from this import d
import argparse
from conrecon.dplearning.adversaries import TrivialTemporalAdversary
from conrecon.utils.graphing import plot_signal_reconstructions
from conrecon.utils.common import create_logger, set_seeds
from conrecon.performance_test_functions import triv_test_entire_file
from conrecon.data.dataset_generation import load_defacto_data

import argparse
import os
import torch
import wandb
from rich import traceback

traceback.install()

wandb_on = False

def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # State stuff here
    ap.add_argument(
        "-e", "--epochs", default=30, help="How many epochs to train for", type=int
    )
    ap.add_argument("--adversary_epochs", default=1, help="How many epochs to train advesrary for", type=int)
    ap.add_argument("--adv_epoch_subsample_percent", default=0.9, help="How many epochs to train advesrary for", type=int)

def main():
    args = argsies()

    triv_test_entire_file(
        test_file,
        args.cols_to_hide,
        trivial_adverary,
        args.episode_length,
        # args.padding_value, # WE NO LONGER USE padding_value
        None, 
        args.batch_size,
        wandb_on=args.wandb
    )
        
if __name__ == "__main__":
    main()
