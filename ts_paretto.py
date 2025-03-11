import argparse
import os
from datetime import datetime

import debugpy
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

from conrecon.data.data_loading import load_defacto_data, split_defacto_runs
from conrecon.dplearning.adversaries import Adversary
from conrecon.dplearning.vae import SequenceToScalarVAE
from conrecon.utils.common import create_logger, set_seeds
from conrecon.performance_test_functions import get_tradeoff_metrics
from conrecon.training_utils import train_vae_and_adversary_bi_level

def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # State stuff here
    ap.add_argument(
        "-e", "--epochs", default=15, help="How many epochs to train for", type=int
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
    ap.add_argument("--batch_size", default=64, type=int)
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
    ap.add_argument("--kl_dig_hypr", "-k", default=0.001, type=float)
    ap.add_argument("--seed", default=0, type=int) 
    ap.add_argument("--debug", "-d", action="store_true", help="Whether or not to use debugpy for trainig") 
    ap.add_argument("--debug_port", default=42022, help="Port to use for debugging") 
    ap.add_argument("--wandb", "-w", action="store_true", help="Whether or not to use wandb for logging") 
    ap.add_argument("--padding_value", default=-1, type=int) 

    return ap.parse_args()

def plot_pareto_frontier(privacies: list[float], utilities: list[float], uvp_tradeoffs: list[float]):
    # Create a publication-ready plot using seaborn
    plt.figure(figsize=(8, 6))  # Standard figure size for paper columns
    
    # Set the style for academic publications
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    
    # Create the main scatter plot with improved aesthetics
    _ = sns.scatterplot(x=privacies, y=utilities, 
                            color='#2E86C1',  # Professional blue color
                            s=100,  # Marker size
                            alpha=0.7)  # Slight transparency
    
    # Add annotations with improved positioning and style
    for i, uvp in enumerate(uvp_tradeoffs):
        plt.annotate(f"UVP: {uvp:.4f}", 
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
    columns, runs_dict, _ = load_defacto_data(args.defacto_data_raw_path)
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
    utility_vs_privacy_tradeoff = np.logspace(-8, 3, 11, base=2)
    # utility_vs_privacy_tradeoff = np.linspace(0.125, 8, 11)

    ########################################
    # Setup up the models
    ########################################
    vae_input_size = len(columns) # I think sending all of them is better
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
    # Configuring Optimizers
    opt_adversary = torch.optim.Adam(model_adversary.parameters(), lr=args.lr)  # type: ignore
    opt_vae = torch.optim.Adam(model_vae.parameters(), lr=args.lr)  # type: ignore

    reset_vae_state = model_vae.state_dict()
    reset_adv_state = model_adversary.state_dict()
    reset_opt_vae_state = opt_vae.state_dict()
    reset_opt_adv_state = opt_adversary.state_dict()


    logger.debug(f"Columns are {columns}")
    logger.debug(f"Runs dict is {runs_dict}")

    time_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger.info(f"Labeling this paretto run as {time_date}")
    run_name = f"uvp_data_{time_date}"
    os.makedirs(f"./results/{run_name}", exist_ok=True)

    privacies = []
    utilities = []
    uvps_so_far = []

    # Prep the data
    # Information comes packed in dictionary elements for each file. We need to mix it up a bit
    all_train_seqs = np.concatenate([seqs for _, seqs in train_seqs.items()], axis=0)
    all_valid_seqs = np.concatenate([seqs for _, seqs in val_seqs.items()], axis=0)
    # Shuffle, Batch, Torch Coversion, Feature Separation
    np.random.shuffle(all_train_seqs)
    np.random.shuffle(all_valid_seqs)
    all_train_seqs = torch.from_numpy(all_train_seqs).to(torch.float32).to(device)
    all_valid_seqs = torch.from_numpy(all_valid_seqs).to(torch.float32).to(device)

    for uvp in utility_vs_privacy_tradeoff:
        # Reset state dict
        model_vae.load_state_dict(reset_vae_state)
        model_adversary.load_state_dict(reset_adv_state)
        opt_vae.load_state_dict(reset_opt_vae_state)
        opt_adversary.load_state_dict(reset_opt_adv_state)
        ########################################
        # Training VAE and Adversary
        ########################################
        logger.info("Starting the VAE Training")
        model_vae, model_adversary, recon_losses, adv_losses = train_vae_and_adversary_bi_level(
            args.batch_size,
            args.cols_to_hide,
            columns,
            all_train_seqs,
            all_valid_seqs,
            None, # We dont need to plot this on every round. Too time consuming
            args.epochs,
            args.adversary_epochs,
            args.adv_intra_epoch_sampling,
            model_vae,
            model_adversary,
            opt_vae,
            opt_adversary,
            args.kl_dig_hypr,
            args.wandb,
            uvp,
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

        # Save these to disk 
        np.save(f"./results/{run_name}/privacies.npy", privacies)
        np.save(f"./results/{run_name}/utilities.npy", utilities)

        uvps_so_far.append(uvp)
        logger.info(f"Validation Metrics are {privacy}, {utility}")
        plot_pareto_frontier(privacies, utilities, uvps_so_far)

    # Save the uvp data
    np.save(f"./results/{run_name}/uvp.npy", utility_vs_privacy_tradeoff)

    logger.info("All baselines complete. Exiting")
    exit()

if __name__ == "__main__":
    logger = create_logger("main_vae")
    main()  
