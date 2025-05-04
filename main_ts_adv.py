import argparse
import os
from math import ceil
from typing import Dict

import debugpy
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich import traceback
from rich.console import Console
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

import wandb
from conrecon.data.data_loading import load_defacto_data, split_defacto_runs
from conrecon.dplearning.adversaries import Adversary, TrivialTemporalAdversary
from conrecon.dplearning.vae import SequenceToOneVectorGeneral
from conrecon.performance_test_functions import (
    advVae_test_file,
)
from conrecon.training_utils import (
    train_adversary,
    train_vae_and_adversary_bi_level,
)
from conrecon.utils.common import create_logger, set_seeds

traceback.install()

console = Console()

wandb_on = False


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    # Data / Misc Paths
    ap.add_argument(
        "--defacto_data_raw_path",
        default="./data/",
        type=str,
        help="Where to load the data from",
    )
    ap.add_argument(
        "--path_figure_dumps",
        default="./figures/recon_only_vae/",
        type=str,
        help="Where to save figures",
    )
    ap.add_argument(
        "--path_data_dumps",
        default="./results/recon_only_vae/",
        type=str,
        help="Where to save results",
    )

    # Hyperparameters
    ap.add_argument(
        "-e", "--epochs", default=30, help="How many epochs to train for", type=int
    )
    ap.add_argument("--batch_size", default=32, type=int)
    ap.add_argument("--rnn_num_layers", default=2, type=int)
    ap.add_argument("--rnn_hidden_size", default=64, type=int)
    ap.add_argument(
        "--cols_to_hide",
        default=[
            5 - 2
        ],  # NOTE: When you work with this code, keep in mind this hardcoded displacement for the private column
        help="Which are the columsn we want no information of",
    )  # Remember 0-index (so 5th)
    ap.add_argument("--vae_latent_output_size", default=64, type=int)
    ap.add_argument("--vae_hidden_size", default=128, type=int)
    ap.add_argument("--episode_length", default=32, type=int)
    ap.add_argument(
        "--splits",
        default={"train_split": 0.8, "val_split": 0.2, "test_split": 0.0},
        type=list,
        nargs="+",
    )
    ap.add_argument(
        "--transformer_num_heads",
        default=3,
        type=int,
        help="Number of Transformer Heads",
    )
    ap.add_argument(
        "--transformer_num_layers",
        default=3,
        type=int,
        help="Number of Tranfomer Layers",
    )
    ap.add_argument(
        "--transformer_dropout", default=0.1, type=float, help="Dropout for transformer"
    )
    ap.add_argument(
        "--adversary_epochs",
        default=2,
        help="How many epochs to train advesrary for",
        type=int,
    )
    ap.add_argument(
        "--clean_adversary_epochs",
        default=20,
        help="How many epochs to train advesrary for",
        type=int,
    )
    ap.add_argument(
        "--adv_epoch_subsample_percent",
        default=1,
        help="How many epochs to train advesrary for",
        type=int,
    )
    ap.add_argument(
        "-c",
        "--priv_utility_tradeoff_coeff",
        default=2,
        # default=4,
        type=float,
        help="The threshold for retaining principal components",
    )
    # ap.add_argument("--kl_dig_hypr", "-k", default=0.001, type=float)
    ap.add_argument("--kl_dig_hypr", "-k", default=0.0009674820321116988, type=float)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--vae_lr", default=0.001, type=float)
    ap.add_argument("--adv_lr", default=0.01, type=float)

    # Data Processing
    ap.add_argument(
        "--oversample_coefficient",
        default=1.6,
        type=float,
        help="The threshold for retaining principal components",
    )
    ap.add_argument(
        "--test_file_name",
        type=str,
        default="run_5.csv",
        help="the one file that will be picked from the rest to test against.",
    )

    # Misc. Logistics
    ap.add_argument(
        "--validation_frequency",
        default=1,
        help="Evaluation Frequency in terms of batches",
    )

    ap.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Whether or not to use debugpy for trainig",
    )
    ap.add_argument(
        "--debug_port",
        default=42022,
        type=int,
        help="Port to attach debugpy to listen to.",
    )
    ap.add_argument(
        "--wandb_on",
        "-w",
        action="store_true",
        help="Whether or not to use wandb for logging",
    )

    args = ap.parse_args()

    if not os.path.exists(".cache/"):
        os.makedirs(".cache/")
    return args
    # Sanity check


def plot_training_losses(recon_losses: list, adv_losses: list, fig_savedest: str):
    os.makedirs(os.path.dirname(fig_savedest), exist_ok=True)
    logger.info("Plotting the training losses")
    _, axs = plt.subplots(2, 1, figsize=(16, 10))
    plt.tight_layout()
    axs[0].plot(recon_losses)
    axs[0].set_title("Reconstruction Loss")
    axs[1].plot(adv_losses)
    axs[1].set_title("Adversary Loss")

    plt.savefig(fig_savedest)
    plt.close()


def plot_single_loss(single_loss: list, title: str, fig_savedest: str):
    os.makedirs(os.path.dirname(fig_savedest), exist_ok=True)
    logger.info(f"Saving figure to {fig_savedest}")
    _, axs = plt.subplots(1, 1, figsize=(16, 8))
    plt.tight_layout()
    axs.plot(single_loss)
    axs.set_title(title)
    plt.savefig(fig_savedest)
    plt.close()


def triv_calculate_validation_metrics(
    all_features: torch.Tensor,
    pub_features_idxs: list[int],
    prv_features_idxs: list[int],
    model_adversary: nn.Module,
) -> Dict[str, float]:
    """
    We use correlation here as our delta-epsilon metric.
    """
    pub_features = all_features[:, :, pub_features_idxs]
    prv_features = all_features[:, :, prv_features_idxs]

    recon_priv = model_adversary(pub_features)

    # Lets just do MSE for now
    prv_mse = torch.mean((prv_features[:, -1, :] - recon_priv) ** 2)

    validation_metrics = {
        "trv_prv_mse": prv_mse.item(),
    }

    return validation_metrics


def baseline_trivial_correlation(
    all_train_seqs: torch.Tensor,
    all_valid_seqs: torch.Tensor,
    prv_features_idxs: list[int],
    batch_size,
    epochs: int,
    lr: float,
    device: torch.device,
):
    """
    Will try to remove a column and simply try to predict it out of the other ones.
    """
    pub_features_idxs = list(
        set(range(all_train_seqs.shape[-1])) - set(prv_features_idxs)
    )

    # Shuffle, Batch, Torch Coversion, Feature Separation
    train_pub = all_train_seqs[:, :, pub_features_idxs]
    train_prv = all_train_seqs[:, :, prv_features_idxs]

    trivial_adversary = TrivialTemporalAdversary(
        num_pub_features=len(pub_features_idxs),
        num_prv_features=len(prv_features_idxs),
        dnn_hidden_size=31,
        rnn_hidden_size=30,
    ).to(device)

    opt_adversary = torch.optim.Adam(trivial_adversary.parameters(), lr=lr)  # type: ignore

    num_batches = ceil(all_train_seqs.shape[0] / batch_size)
    # Once we have that we can start training the adversary
    adv_losses = []
    for e in tqdm(range(epochs), desc="Epochs"):
        for batch_no in tqdm(range(num_batches), desc="Batches"):
            batch_pub = train_pub[batch_no * batch_size : (batch_no + 1) * batch_size]
            batch_prv = train_prv[batch_no * batch_size : (batch_no + 1) * batch_size]
            if batch_pub.shape[0] != batch_size:
                continue

            ########################################
            # 1. Get the adversary to guess the sensitive column
            ########################################
            # Get the latent features and sanitized data
            adversary_guess_flat = trivial_adversary(batch_pub)

            # Check on performance
            batch_y_flat = batch_prv[:, -1, :].view(
                -1, batch_prv.shape[-1]
            )  # Grab only last in sequeence
            adv_train_loss = F.mse_loss(adversary_guess_flat, batch_y_flat)
            trivial_adversary.zero_grad()
            adv_train_loss.backward()
            opt_adversary.step()

            adv_losses.append(adv_train_loss.item())

            if wandb_on:
                wandb.log(
                    {
                        "triv_adv_train_loss": adv_train_loss.item(),
                    }
                )

            if batch_no % 16 == 0:
                # TODO: Finish the validation implementaion with correlation
                # - Log the validation metrics here
                trivial_adversary.eval()
                with torch.no_grad():
                    validation_metrics = triv_calculate_validation_metrics(
                        all_valid_seqs,
                        pub_features_idxs,
                        prv_features_idxs,
                        trivial_adversary,
                    )
                trivial_adversary.train()

                # Report to wandb
                if wandb_on:
                    wandb.log(validation_metrics)
    return trivial_adversary, adv_losses


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(args.seed)
    logger = create_logger("main_training")

    if args.wandb_on:
        wandb.init(project="private_control_pure_vae", config=vars(args))

    if args.debug:
        logger.info("\033[1;33m Waiting for debugger to attach...\033[0m")
        debugpy.listen(("0.0.0.0", args.debug_port))
        debugpy.wait_for_client()

    ########################################
    # Data Loading
    ########################################
    # Get a dictionary where each key is a run/file and each value is the data for that run
    runs_dict, columns = load_defacto_data(args.defacto_data_raw_path)

    # DEBUG: remove this afterwards
    file_names = "\n".join([f"\t-{file_name}" for file_name in runs_dict.keys()])
    print(f"Show me all files being used:\n {file_names}")

    # Separate them into their splits (and also interpolate)
    train_seqs, val_seqs, test_file = split_defacto_runs(
        run_dict=runs_dict,
        train_split=args.splits["train_split"],
        test_file_name=args.test_file_name,
        val_split=args.splits["val_split"],
        seq_length=args.episode_length,
        oversample_coefficient=args.oversample_coefficient,  # Scale
        scale=True,
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
    non_0_idxs = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    test_file = test_file[:, non_0_idxs]
    all_train_seqs = all_train_seqs[:, :, non_0_idxs]
    all_valid_seqs = all_valid_seqs[:, :, non_0_idxs]

    # Shuffle, Batch, Torch Coversion, Feature Separation
    np.random.shuffle(all_train_seqs)
    np.random.shuffle(all_valid_seqs)
    all_train_seqs = torch.from_numpy(all_train_seqs).to(torch.float32).to(device)
    all_valid_seqs = torch.from_numpy(all_valid_seqs).to(torch.float32).to(device)

    num_columns = all_train_seqs.shape[-1]
    print(f"num_columns is {num_columns}")
    num_private_cols = len(args.cols_to_hide)
    num_public_cols = num_columns - num_private_cols
    print(f"num_public_cols is {num_public_cols}")

    ########################################
    # Setup up the models
    ########################################
    s2one_input_size = num_columns
    # TODO: Get the model going
    model_vae = SequenceToOneVectorGeneral(
        s2one_input_size=s2one_input_size,
        num_sanitized_features=num_public_cols,
        vae_latent_output_size=args.vae_latent_output_size,
        vae_hidden_size=args.vae_hidden_size,
        seq_processor_type="lstm",  # Options: "lstm", "bilstm", "transformer"
        rnn_num_layers=args.rnn_num_layers,
        rnn_hidden_size=args.rnn_hidden_size,
        transformer_num_heads=args.transformer_num_heads,
        transformer_num_layers=args.transformer_num_layers,
        transformer_dropout=args.transformer_dropout,
        max_seq_length=args.episode_length,
    ).to(device)
    model_adversary = Adversary(
        input_size=args.vae_latent_output_size,
        hidden_size=args.vae_hidden_size,
        num_classes=num_private_cols,
    ).to(device)

    ########################################
    # Training VAE and Adversary
    ########################################
    logger.info("Starting the VAE Training")

    # Get the optimziers
    opt_vae = torch.optim.Adam(model_vae.parameters(), lr=args.vae_lr)  # type: ignore
    opt_adv = torch.optim.Adam(model_adversary.parameters(), lr=args.adv_lr) # type:ignore

    model_vae, model_adversary, recon_losses = train_vae_and_adversary_bi_level(
        batch_size=args.batch_size,
        prv_columns=args.cols_to_hide,
        all_train_seqs=all_train_seqs,
        all_validation_seqs=all_valid_seqs,
        epochs=args.epochs,
        model_vae=model_vae,
        model_adversary=model_adversary,
        adv_train_subepochs=args.adversary_epochs,
        adv_epoch_subsample_percent=args.adv_epoch_subsample_percent,
        optimizer_vae=opt_vae,
        optimizer_adv=opt_adv,
        kl_dig_hypr=args.kl_dig_hypr,
        wandb_on=args.wandb_on,
        priv_utility_tradeoff_coeff=args.priv_utility_tradeoff_coeff,
        validate_every_n_batches=args.validation_frequency,
    )

    logger.info("Running a final clean training of the adversary...")
    clean_model_adversary = Adversary(
        input_size=args.vae_latent_output_size,
        hidden_size=args.vae_hidden_size,
        num_classes=num_private_cols,
    ).to(device)
    clean_opt_adv  = torch.optim.Adam(clean_model_adversary.parameters(), lr=args.adv_lr) # type:ignore
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

    ########################################
    # Finally, we do visual Validation
    ########################################

    # TODO: PLot recon losses
    if wandb_on:
        wandb.log({"recon_losses": recon_losses})
    logger.info(f"Validation Metrics are {metrics}")

    ########################################
    ## Benchmarks
    #
    # Training Trivial Adversary
    ########################################
    # trivial_adverary, adv_losses = baseline_trivial_correlation(
    #     all_train_seqs,
    #     all_valid_seqs,
    #     args.cols_to_hide,
    #     args.batch_size,
    #     args.epochs,
    #     args.lr,
    #     device,
    # )
    # plot_single_loss(
    #     adv_losses,
    #     "Trivial Adversary Losses",
    #     "./figures/new_data_vae/trivial-adv_losses.png",
    # )
    # triv_test_entire_file(
    #     test_file,
    #     args.cols_to_hide,
    #     trivial_adverary,
    #     args.episode_length,
    #     # args.padding_value, # WE NO LONGER USE padding_value
    #     None,
    #     args.batch_size,
    #     wandb_on=args.wandb_on,
    # )

    # Need to see this
    # privacy, utility = get_tradeoff_metrics(
    #     test_file,
    #     args.cols_to_hide,
    #     model_vae,
    #     model_adversary,
    #     args.episode_length,
    #     # args.padding_value, # WE NO LONGER USE padding_value
    #     None,
    #     args.batch_size,
    # )


if __name__ == "__main__":
    logger = create_logger("main_vae")
    main()
