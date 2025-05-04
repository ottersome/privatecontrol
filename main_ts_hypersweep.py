import torch 
import numpy as np

import wandb
from conrecon.data.data_loading import load_defacto_data, split_defacto_runs
from conrecon.dplearning.adversaries import Adversary
from conrecon.dplearning.vae import SequenceToScalarVAE
from conrecon.performance_test_functions import get_tradeoff_metrics
from conrecon.training_utils import train_vae_and_adversary_bi_level
from conrecon.utils.common import create_logger, set_seeds
from main_ts import plot_training_losses

wandb.login()

sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        # "adversary_epochs": {"values": [1, 2, 3]},
        "adversary_epochs": {"values": [1]},
        # "batch_size": {"values": [32, 64, 128]},
        "batch_size": {"values": [32]},
        # "rnn_num_layers": {"values": [1, 2]},
        "rnn_num_layers": {"values": [1]},
        # "rnn_hidden_size": {"values": [32, 64]},
        "rnn_hidden_size": {"values": [32]},
        "vae_latent_size": {"values": [64]},
        "vae_hidden_size": {"values": [32, 64, 128, 256]},
        "kl_dig_hypr": {"max": 10.0, "min": 0.0001},
        "priv_utility_tradeoff_coeff": {"max": 16.0, "min": 0.0001},
    },
}
# sweep_configuration = {
# sweep_configuration = {
#     "method": "random",
#     "metric": {"goal": "minimize", "name": "score"},
#     "parameters": {
#         "x": {"max": 0.1, "min": 0.01},
#         "y": {"values": [1, 3, 7]},
#     },
# }

EPOCHS = 25
EPISODE_LENGTH = 32
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2
DEFACTO_DATA_RAW_PATH = "./data/"
ADV_EPOCH_SUBSAMPLE_PERCENT =  0.9
OVERSAMPLE_COEFFICIENT=1.6
COLS_TO_HIDE = [2]
LR = 0.001

def main():
    wandb.init(project="private_control_sweep")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(0)
    logger = create_logger("main_training")

    wandb_args = wandb.config

    wandb.init(project="private_control")

    columns, runs_dict, debug_file = load_defacto_data(DEFACTO_DATA_RAW_PATH)

    # Separate them into their splits (and also interpolate)
    train_seqs, val_seqs, test_file = split_defacto_runs(
        run_dict=runs_dict,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        seq_length=EPISODE_LENGTH,
        oversample_coefficient=OVERSAMPLE_COEFFICIENT,
        scale=True,  # Scale
    )

    logger.info(f"Using device is {device}")

    # Get Informaiton for the VAE
    logger.debug(f"Columns are {columns}")
    logger.debug(f"Runs dict is {runs_dict}")

    # Prep the data
    # Information comes packed in dictionary elements for each file. We need to mix it up a bit
    all_train_seqs = np.concatenate([seqs for _, seqs in train_seqs.items()], axis=0)
    all_valid_seqs = np.concatenate([seqs for _, seqs in val_seqs.items()], axis=0)

    print(f"Shape of all_train_seqs is {all_train_seqs.shape}")
    print(f"Shape of test_file is {test_file.shape}")
    non_0_idxs = [0,3,4,5,6,7,8,9,10,11,12,13,14,15]
    test_file = test_file[:,non_0_idxs]
    all_train_seqs = all_train_seqs[:,:,non_0_idxs]
    
    # Shuffle, Batch, Torch Coversion, Feature Separation
    np.random.shuffle(all_train_seqs)
    np.random.shuffle(all_valid_seqs)
    all_train_seqs = torch.from_numpy(all_train_seqs).to(torch.float32).to(device)
    all_valid_seqs = torch.from_numpy(all_valid_seqs).to(torch.float32).to(device)

    num_columns = all_train_seqs.shape[-1]
    print(f"num_columns is {num_columns}")
    num_private_cols = len(COLS_TO_HIDE)
    num_public_cols = num_columns - num_private_cols
    print(f"num_public_cols is {num_public_cols}")

    ########################################
    # Setup up the models
    ########################################
    vae_input_size = num_columns
    # TODO: Get the model going
    model_vae = SequenceToScalarVAE(
        input_size=vae_input_size,
        num_sanitized_features=num_public_cols,
        latent_size=wandb_args.vae_latent_size,
        hidden_size=wandb_args.vae_hidden_size,
        rnn_num_layers=wandb_args.rnn_num_layers,
        rnn_hidden_size=wandb_args.rnn_hidden_size,
    ).to(device)

    model_adversary = Adversary(
        input_size=wandb_args.vae_latent_size,
        hidden_size=wandb_args.vae_hidden_size,
        num_classes=num_private_cols,
    ).to(device)

    # Configuring Optimizers
    opt_adversary = torch.optim.Adam(model_adversary.parameters(), lr=LR)  # type: ignore
    opt_vae = torch.optim.Adam(model_vae.parameters(), lr=LR)  # type: ignore


    ########################################
    # Training VAE and Adversary
    ########################################
    logger.info("Starting the VAE Training")
    # THis reports its loss metrics within itself
    model_vae, model_adversary, recon_losses, adv_losses = train_vae_and_adversary_bi_level(
        wandb_args.batch_size,
        COLS_TO_HIDE,
        all_train_seqs,
        all_valid_seqs,
        None, # TOREM: THis is for making it faster fow now
        EPOCHS,
        wandb_args.adversary_epochs,
        ADV_EPOCH_SUBSAMPLE_PERCENT,
        model_vae,
        model_adversary,
        opt_vae,
        opt_adversary,
        wandb_args.kl_dig_hypr,
        True,
        wandb_args.priv_utility_tradeoff_coeff,
    )

    # Need to see this
    privacy, utility = get_tradeoff_metrics(
        test_file,
        COLS_TO_HIDE,
        model_vae,
        model_adversary,
        EPISODE_LENGTH,
        # args.padding_value, # WE NO LONGER USE padding_value
        None,
        wandb_args.batch_size,
    )
    logger.info(f"Final Validation Metrics are {privacy}, {utility}")
    wandb.log({
        "Final Privacy": privacy,
         "Final Utility": utility,
    })


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="private_control_sweep")
    wandb.agent(sweep_id, function=main, count=20)
