import argparse
import os
from datetime import datetime

import debugpy
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

from conrecon.dplearning.adversaries import Adversary
from conrecon.utils.common import create_logger, set_seeds
from conrecon.performance_test_functions import get_tradeoff_metrics
from conrecon.training_utils import train_adversary
from main_ts_adv import main_data_prep, main_model_loading, main_training_run

def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # State stuff here

    # Data / Misc Paths
    ap.add_argument(
        "--defacto_data_raw_path",
        default="./data/",
        type=str,
        help="Where to load the data from",
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
        "--num_uvp_samples",
        default=11,
        type=int,
        help="How many uvp samples to take",
    )

    # ---------------------------------------------------
    ap.add_argument("--debug", "-d", action="store_true", help="Whether or not to use debugpy for trainig") 
    ap.add_argument("--debug_port", default=42022, help="Port to use for debugging") 
    ap.add_argument("--wandb", "-w", action="store_true", help="Whether or not to use wandb for logging") 

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



    ########################################
    # Hyperparemters
    ########################################
    # TODO: vary this boi
    # utility_vs_privacy_tradeoff = np.linspace(0.0, 10, 10) 
    # Do logspace instead
    utility_vs_privacy_tradeoff = np.logspace(-7, 1, args.num_uvp_samples, base=2)
    # utility_vs_privacy_tradeoff = np.linspace(0.125, 8, 11)

    ########################################
    # Loading up Data
    ########################################
    all_train_seqs, all_valid_seqs, test_file =  main_data_prep(
        cols_to_hide=args.cols_to_hide,
        defacto_data_raw_path=args.defacto_data_raw_path,
        device=device,
        episode_length=args.episode_length,
        oversample_coefficient=args.oversample_coefficient,
        splits=args.splits,
        test_file_name=args.test_file_name,
    )
    num_columns = all_train_seqs.shape[-1]
    num_private_cols = len(args.cols_to_hide)
    num_public_cols = num_columns - num_private_cols

    ########################################
    # Setup up the models
    ########################################
    model_vae, model_adversary, opt_adv, opt_vae = main_model_loading(
            adv_lr=args.adv_lr,
            device=device,
            episode_length=args.episode_length,
            num_columns=num_columns,
            num_private_cols=num_private_cols,
            num_public_cols=num_public_cols,
            rnn_hidden_size=args.rnn_hidden_size,
            rnn_num_layers=args.rnn_num_layers,
            transformer_dropout=args.transformer_dropout,
            transformer_num_heads=args.transformer_num_heads,  # type: ignore
            transformer_num_layers=args.transformer_num_layers,  # type: ignore
            vae_hidden_size=args.vae_hidden_size,  # type: ignore
            vae_latent_output_size=args.vae_latent_output_size,  # type: ignore
            vae_lr=args.vae_lr,  # type: ignore
    )

    #########################################
    # Remember Initial States
    ########################################
    reset_vae_state = model_vae.state_dict()
    reset_adv_state = model_adversary.state_dict()
    reset_opt_vae_state = opt_vae.state_dict()
    reset_opt_adv_state = opt_adv.state_dict()

    time_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger.info(f"Labeling this paretto run as {time_date}")
    run_name = f"uvp_data_{time_date}"
    os.makedirs(f"./results/{run_name}", exist_ok=True)

    privacies = []
    utilities = []
    uvps_so_far = []

    for uvp in utility_vs_privacy_tradeoff:
        # Reset state dict
        model_vae.load_state_dict(reset_vae_state)
        model_adversary.load_state_dict(reset_adv_state)
        opt_vae.load_state_dict(reset_opt_vae_state)
        opt_adv.load_state_dict(reset_opt_adv_state)
        ########################################
        # Training VAE and Adversary
        ########################################
        logger.info("Starting the VAE Training")
        model_vae, recon_losses = main_training_run(
            adv_epoch_subsample_percent=args.adv_epoch_subsample_percent,
            adversary_epochs=args.adversary_epochs,
            all_train_seqs=all_train_seqs,
            all_valid_seqs=all_valid_seqs,
            batch_size=args.batch_size,
            cols_to_hide=args.cols_to_hide,
            epochs=args.epochs,
            kl_dig_hypr=args.kl_dig_hypr,
            model_adversary=model_adversary,
            model_vae=model_vae,
            opt_adv=opt_adv,
            opt_vae=opt_vae,
            priv_utility_tradeoff_coeff=uvp,  # type: ignore
            validation_frequency=args.validation_frequency,  # type: ignore
            wandb_on=False
        )

        #########################################
        # Evaluation
        ########################################
        # First Need to get a clean adversary
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
        privacy, utility = get_tradeoff_metrics(
            test_file,
            args.cols_to_hide,
            model_vae,
            clean_model_adversary,
            args.episode_length,
        )
        privacies.append(privacy)
        utilities.append(utility)

        # Save these to disk 
        np.save(f"./results/{run_name}/privacies.npy", privacies)
        np.save(f"./results/{run_name}/utilities.npy", utilities)

        uvps_so_far.append(uvp)
        logger.info(f"Validation Metrics are privacy: {privacy}, utility: {utility}")
        logger.info(f"Privacies saved to {f'./results/{run_name}/privacies.npy'}")
        logger.info(f"Utilities saved to {f'./results/{run_name}/utilities.npy'}")
        plot_pareto_frontier(privacies, utilities, uvps_so_far)

    # Save the uvp data
    np.save(f"./results/{run_name}/uvp.npy", utility_vs_privacy_tradeoff)

    logger.info("All baselines complete. Exiting")
    exit()

if __name__ == "__main__":
    logger = create_logger("main_vae")
    main()  
