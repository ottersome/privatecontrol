import debugpy
import numpy as np
import torch
import wandb
import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from conrecon.data.data_loading import load_defacto_data, split_defacto_runs
from conrecon.dplearning.adversaries import Adversary
from conrecon.dplearning.vae import SequenceToScalarVAE
from conrecon.utils import create_logger, set_seeds
from conrecon.performance_test_functions import get_tradeoff_metrics
from conrecon.training_utils import train_vae_and_adversary, train_vae_and_adversary_bi_level

def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # State stuff here
    ap.add_argument(
        "-e", "--epochs", default=100, help="How many epochs to train for", type=int
    )
    ap.add_argument("-l", "--lr", default=0.001, type=float) 
    ap.add_argument("--adversary_epochs", default=3, help="How many epochs to train advesrary for", type=int)
    ap.add_argument("--train_val_data_sample_coeff", default=1.8, help="How much to sample both training and validation data. Mostly for oversampling.", type=int)
    ap.add_argument("--adv_intra_epoch_sampling", default=1.0, help="What percentage of batches to take within the adversary training loop.", type=int)
    ap.add_argument(
        "--defacto_data_raw_path",
        default="./data/",
        type=str,
        help="Where to load the data from",
    )
    ap.add_argument("--batch_size", default=16, type=int)
    ap.add_argument("--rnn_num_layers", default=2, type=int)
    ap.add_argument("--rnn_hidden_size", default=15, type=int)
    ap.add_argument(
        "--cols_to_hide",
        default=[5],
        help="Which are the columsn we want no information of",
    )  # Remember 0-index (so 5th)
    ap.add_argument("--vae_latent_size", default=32, type=int)
    ap.add_argument("--episode_length", default=32, type=int)
    ap.add_argument("--episode_gap", default=3, type=int)
    ap.add_argument("--vae_hidden_size", default=32, type=int)
    ap.add_argument(
        "--splits",
        default={"train_split": 0.8, "val_split": 0.2, "test_split": 0.0},
        type=list,
        nargs="+",
    )
    ap.add_argument(
        "--priv_utility_tradeoff_coeff",
        default=1.1,
        # default=4,
        type=float,
        help="The threshold for retaining principal components",
    )
    ap.add_argument("--kl_dig_hypr", "-k", default=0.001, type=float) # type: ignore 
    ap.add_argument("--seed", default=0, type=int) 
    ap.add_argument("--debug", "-d", action="store_true", help="Whether or not to use debugpy for trainig") 
    ap.add_argument("--debug_port", default=42020, help="Port to use for debugging") 
    ap.add_argument("--wandb", "-w", action="store_true", help="Whether or not to use wandb for logging") 
    ap.add_argument("--padding_value", default=-1, type=int) 

    return ap.parse_args()

def main():
    args = argsies()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(args.seed)
    logger = create_logger("main_training")

    if args.debug:
        logger.info("\033[1;33m Waiting for debugger to attach...\033[0m")
        debugpy.listen(("0.0.0.0", args.debug_port))
        debugpy.wait_for_client()

    # TODO: Make it  so that generate_dataset checks if params are the same
    columns, runs_dict, debug_file = load_defacto_data(args.defacto_data_raw_path)
    num_columns = len(columns)
    num_private_cols = len(args.cols_to_hide)
    num_public_cols = num_columns - num_private_cols

    # Separate them into their splits (and also interpolate)
    train_seqs, val_seqs, test_file = split_defacto_runs(
        runs_dict,
        args.splits["train_split"],
        args.splits["val_split"],
        args.episode_length,
        args.train_val_data_sample_coeff,
        True, # Scale
    )

    ########################################
    # Hyperparemters
    ########################################
    # TODO: vary this boi
    # utility_vs_privacy_tradeoff = np.linspace(0.0, 10, 10) 
    # Do logspace instead
    utility_vs_privacy_tradeoff = np.logspace(-3, 3, 11, base=2)

    ########################################
    # Setup up the models
    ########################################
    vae_input_size = len(columns) # I think sending all of them is better
    # TODO: Get the model going

    logger.debug(f"Columns are {columns}")
    logger.debug(f"Runs dict is {runs_dict}")

    privacies = []
    utilities = []
    for uvp in utility_vs_privacy_tradeoff:
        ########################################
        # Training VAE and Adversary
        ########################################
        model_vae = SequenceToScalarVAE(
            input_size=vae_input_size,
            num_sanitized_features=num_public_cols,
            latent_size=args.vae_latent_size,
            hidden_size=args.vae_hidden_size,
            rnn_num_layers=args.rnn_num_layers,
            rnn_hidden_size=args.rnn_hidden_size,
        ).to(device)
        model_adversary = Adversary(
            input_size=args.vae_latent_size,
            hidden_size=args.vae_hidden_size,
            num_classes=num_private_cols,
        ).to(device)

        logger.info("Starting the VAE Training")
        model_vae, model_adversary, recon_losses, adv_losses = train_vae_and_adversary_bi_level(
            args.batch_size,
            args.cols_to_hide,
            columns,
            device,
            train_seqs,
            val_seqs,
            args.epochs,
            args.adversary_epochs,
            args.adv_intra_epoch_sampling,
            model_vae,
            model_adversary,
            args.lr,
            args.kl_dig_hypr,
            args.wandb,
            logger,
            args.priv_utility_tradeoff_coeff,
        )
        privacy, utility = get_tradeoff_metrics(
            test_file,
            args.cols_to_hide,
            model_vae,
            model_adversary,
            args.episode_length,
            args.padding_value,
            args.batch_size,
        )
        privacies.append(privacy)
        utilities.append(utility)
        logger.info(f"Validation Metrics are {privacy}, {utility}")

    # Create a publication-ready plot using seaborn
    plt.figure(figsize=(8, 6))  # Standard figure size for paper columns
    
    # Set the style for academic publications
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    
    # Create the main scatter plot with improved aesthetics
    scatter = sns.scatterplot(x=privacies, y=utilities, 
                            color='#2E86C1',  # Professional blue color
                            s=100,  # Marker size
                            alpha=0.7)  # Slight transparency
    
    # Add annotations with improved positioning and style
    for i, uvp in enumerate(utility_vs_privacy_tradeoff):
        plt.annotate(f"UVP: {uvp:.2f}", 
                    (privacies[i], utilities[i]),
                    xytext=(8, 8),
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
    
    # Customize the plot with publication-quality formatting
    plt.title("Privacy-Utility Trade-off Analysis", pad=20)
    plt.xlabel("Privacy Score", labelpad=10)
    plt.ylabel("Utility Score", labelpad=10)
    
    # Adjust layout to prevent label clipping
    plt.tight_layout()
    
    # Save with high DPI for print quality
    plt.savefig("./figures/privacy_vs_utility.png", 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white')
    plt.close()
    logger.info("All baselines complete. Exiting")
    exit()

if __name__ == "__main__":
    logger = create_logger("main_vae")
    main()  
