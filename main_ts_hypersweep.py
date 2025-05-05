import torch 
import numpy as np

import wandb
from conrecon.data.data_loading import load_defacto_data, split_defacto_runs
from conrecon.dplearning.adversaries import Adversary
from conrecon.dplearning.vae import SequenceToScalarVAE
from conrecon.performance_test_functions import get_tradeoff_metrics, triv_test_entire_file
from conrecon.training_utils import train_vae_and_adversary_bi_level
from conrecon.utils.common import create_logger, set_seeds
from main_ts import plot_training_losses
from main_ts_adv import main_data_prep, main_model_loading, main_training_run

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
        "adv_lr": {"max": 0.1, "min": 0.0001},
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
COLS_TO_HIDE = [3]
LR = 0.001
TEST_FILE_NAME = "run_5.csv"
CLEAN_ADVERSARY_EPOCHS = 20

def main():
    wandb.init(project="private_control_sweep")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(0)
    logger = create_logger("main_training")

    wandb_args = wandb.config

    wandb.init(project="private_control")

    # Load the data
    all_train_seqs, all_valid_seqs, test_file = main_data_prep(
        cols_to_hide=COLS_TO_HIDE,
        defacto_data_raw_path=DEFACTO_DATA_RAW_PATH,
        device=device,
        episode_length=EPISODE_LENGTH,
        oversample_coefficient=OVERSAMPLE_COEFFICIENT,
        splits={"train_split": TRAIN_SPLIT, "val_split": VAL_SPLIT, "test_split": 0.0},
        test_file_name=TEST_FILE_NAME,
    )
    num_columns = all_train_seqs.shape[-1]
    num_private_cols = len(COLS_TO_HIDE)
    num_public_cols = num_columns - num_private_cols

    ########################################
    # Setup up the models
    ########################################
    main_vae, main_adversary, opt_adv, opt_vae = main_model_loading(
            adv_lr=wandb_args.adv_lr,
            device=device,
            episode_length=EPISODE_LENGTH,
            num_columns=num_columns,    
            num_private_cols=num_private_cols,
            num_public_cols=num_public_cols,
            rnn_hidden_size=wandb_args.rnn_hidden_size,
            rnn_num_layers=wandb_args.rnn_num_layers,
            transformer_dropout=wandb_args.transformer_dropout,
            transformer_num_heads=wandb_args.transformer_num_heads,
            transformer_num_layers=wandb_args.transformer_num_layers,
            vae_hidden_size=wandb_args.vae_hidden_size,
            vae_latent_output_size=wandb_args.vae_latent_size,
            vae_lr=wandb_args.vae_lr,
    )

    ########################################
    # Training VAE and Adversary
    ########################################
    logger.info("Starting the VAE Training")
    # THis reports its loss metrics within itself
    model_vae, recon_losses = main_training_run(
            adv_epoch_subsample_percent=wandb_args.adv_epoch_subsample_percent,
            adversary_epochs=wandb_args.adversary_epochs,
            all_train_seqs=all_train_seqs,
            all_valid_seqs=all_valid_seqs,
            batch_size=wandb_args.batch_size,
            cols_to_hide=COLS_TO_HIDE,
            epochs=EPOCHS,
            kl_dig_hypr=wandb_args.kl_dig_hypr,
            model_adversary=main_adversary,
            model_vae=main_vae,
            opt_adv=opt_adv,
            opt_vae=opt_vae,
            priv_utility_tradeoff_coeff=wandb_args.priv_utility_tradeoff_coeff,
            validation_frequency=wandb_args.validation_frequency,
            wandb_on=True
    )

    ########################################
    # Evaluating 
    ########################################
    logger.info("Running a final clean training of the adversary...")
    num_private_cols = len(COLS_TO_HIDE)
    clean_model_adversary = Adversary(
        input_size=wandb_args.vae_latent_output_size,
        hidden_size=wandb_args.vae_hidden_size,
        num_classes=num_private_cols,
    ).to(device)
    clean_opt_adv  = torch.optim.Adam(clean_model_adversary.parameters(), lr=wandb.adv_lr) # type:ignore
    train_adversary(
        model_vae,
        clean_model_adversary,
        clean_opt_adv,
        args.clean_adversary_epochs,
        1,
        all_train_seqs,
        args.cols_to_hide,
        args.batch_size,
    )
    metrics = advVae_test_file(
        test_file,
        args.cols_to_hide,
        model_vae,
        clean_model_adversary,
        args.episode_length,
        batch_size=args.batch_size,
        wandb_on=args.wandb_on,
    )
    privacy, utility = get_tradeoff_metrics(
        test_file,
        args.cols_to_hide,
        model_vae,
        clean_model_adversary,
        args.episode_length,
    )
    # # Need to see this
    # privacy, utility = get_tradeoff_metrics(
    #     test_file,
    #     COLS_TO_HIDE,
    #     model_vae,
    #     model_adversary,
    #     EPISODE_LENGTH,
    #     # args.padding_value, # WE NO LONGER USE padding_value
    #     None,
    #     wandb_args.batch_size,
    # )
    logger.info(f"Final Validation Metrics are {privacy}, {utility}")
    wandb.log({
        "Final Privacy": privacy,
         "Final Utility": utility,
    })


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="private_control_sweep")
    wandb.agent(sweep_id, function=main, count=20)
