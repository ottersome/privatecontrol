import argparse
import os
from math import ceil
from typing import Dict, List, OrderedDict, Tuple

import debugpy
import matplotlib.pyplot as plt
import numpy as np
import torch
from rich import traceback
from rich.console import Console
from rich.live import Live
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import wandb
from sklearn.decomposition import PCA

from conrecon.data.data_loading import load_defacto_data, split_defacto_runs
from conrecon.data.dataset_generation import collect_n_sequential_batches, spot_backhistory
from conrecon.dplearning.adversaries import Adversary, TrivialTemporalAdversary, PCATemporalAdversary
from conrecon.dplearning.vae import SequenceToScalarVAE, SequenceToScalarVAE
from conrecon.plotting import TrainLayout
from conrecon.utils import create_logger, set_seeds
from conrecon.validation_functions import calculate_validation_metrics
from conrecon.performance_test_functions import test_entire_file, triv_test_entire_file, pca_test_entire_file

traceback.install()

console = Console()

wandb_on = False

def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # State stuff here
    ap.add_argument(
        "-e", "--epochs", default=8, help="How many epochs to train for", type=int
    )
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
        default=[4],
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
    ap.add_argument("--kl_dig_hypr", default=0.1, type=float)

    ap.add_argument("--no-autoindent")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--lr", default=0.01, type=float)
    ap.add_argument("--first_n_states", default=7, type=int)
    ap.add_argument("--adversary_hidden_size", default=32, type=int)
    ap.add_argument("--padding_value", default=-1, type=int)

    ap.add_argument("--vae_ds_cache", default=".cache/pykalpkg_vaeds.csv", type=str)
    ap.add_argument("--ds_cache", default=".cache/pykalpkg_ds.csv", type=str)
    ap.add_argument(
        "--saveplot_dest",
        default="./figures/pykalman_transformer/",
        help="Where to save the outputs",
    )

    ap.add_argument(
        "--eval_interval", default=100, help="How many epochs to train for", type=int
    )
    ap.add_argument(
        "--eval_size", default=4, help="How many systems to generate", type=int
    )

    ap.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Whether or not to use debugpy for trainig",
    )
    ap.add_argument(
        "--debug_port",
        default=42020,
        type=int,
        help="Port to attach debugpy to listen to.",
    )
    ap.add_argument(
        "--shuffle",
        action="store_false",
        help="Whether or not to shuffle the data",
    )
    ap.add_argument(
        "--wandb",
        "-w",
        action="store_true",
        help="Whether or not to use wandb for logging",
    )
    ap.add_argument(
        "--correlation_threshold",
        default=0.1,
        type=float,
        help="The threshold for retaining principal components",
    )

    args = ap.parse_args()

    if not os.path.exists(args.saveplot_dest):
        os.makedirs(args.saveplot_dest)
    if not os.path.exists(".cache/"):
        os.makedirs(".cache/")
    return args
    # Sanity check

def compare_reconstruction():
    """
    Will take the original set of features and
    """
    raise NotImplementedError

def plot_training_losses(recon_losses: List, adv_losses: List, fig_savedest: str):
    logger.info("Plotting the training losses")
    fig, axs = plt.subplots(1, 2, figsize=(16,10))
    axs[0].plot(recon_losses)
    axs[0].set_title("Reconstruction Loss")
    axs[1].plot(adv_losses)
    axs[1].set_title("Adversary Loss")

    plt.savefig(fig_savedest)
    plt.close()

def train_v1(
    batch_size: int,
    prv_columns: List[int],
    data_columns: List[str],
    device: torch.device,
    ds_train: Dict[str, np.ndarray],
    ds_val: Dict[str, np.ndarray],
    epochs: int,
    model_vae: SequenceToScalarVAE,
    model_adversary: Adversary,
    learning_rate: float,
    kl_dig_hypr: float,
) -> Tuple[nn.Module, nn.Module, List, List]:
    """
    Training Loop
        ds_train: np.ndarray (num_batches, batch_size, features),
    """

    pub_columns = list(set(range(len(data_columns))) - set(prv_columns))
    device = next(model_vae.parameters()).device

    # Configuring Optimizers
    opt_adversary = torch.optim.Adam(model_adversary.parameters(), lr=learning_rate)  # type: ignore
    opt_vae = torch.optim.Adam(model_vae.parameters(), lr=learning_rate)  # type: ignore

    ##  A bit of extra processing of data. (Specific to this version of training)
    # Information comes packed in dictionary elements for each file. We need to mix it up a bit
    all_train_seqs = np.concatenate([ seqs for _, seqs in ds_train.items()], axis=0)
    all_valid_seqs = np.concatenate([ seqs for _, seqs in ds_val.items()], axis=0)
    # Shuffle, Batch, Torch Coversion, Feature Separation
    np.random.shuffle(all_train_seqs)
    np.random.shuffle(all_valid_seqs)
    batch_amnt  = all_train_seqs.shape[0] // batch_size
    # all_train_seqs = all_train_seqs.reshape(batch_amnt, batch_size, all_train_seqs.shape[-2], all_train_seqs.shape[-1])
    # all_valid_seqs = all_valid_seqs.reshape(batch_amnt, batch_size, all_valid_seqs.shape[-2], all_valid_seqs.shape[-1])
    all_train_seqs = torch.from_numpy(all_train_seqs).to(torch.float32).to(device)
    all_valid_seqs = torch.from_numpy(all_valid_seqs).to(torch.float32).to(device)
    train_pub = all_train_seqs[:,:,pub_columns]
    train_prv = all_train_seqs[:,:,prv_columns]

    num_batches     = ceil(all_train_seqs.shape[0] / batch_size)
    sequence_len    = all_train_seqs.shape[1]

    ########################################
    # Get Batches
    ########################################
    logger.info(f"Working with {num_batches} num_batches, each with size:  {batch_size}, and sequence/episode length {sequence_len}")
    recon_losses = []
    adv_losses = []
    for e in tqdm(range(epochs), desc="Epochs"):
        logger.info(f"Epoch {e} of {epochs}")
        for batch_no in tqdm(range(num_batches), desc="Batches"):
            # Now Get the new VAE generations
            batch_all = all_train_seqs[batch_no * batch_size : (batch_no + 1) * batch_size]
            batch_pub = train_pub[batch_no * batch_size : (batch_no + 1) * batch_size]
            batch_prv = train_prv[batch_no * batch_size : (batch_no + 1) * batch_size]
            if batch_pub.shape[0] != batch_size:
                continue

            logger.info(f"Batch {batch_no} out of {num_batches} with shape {batch_pub.shape}")

            ########################################
            # 1. Get the adversary to guess the sensitive column
            ########################################
            # Get the latent features and sanitized data
            latent_z, sanitized_data, _ = model_vae(batch_all)

            # Take Latent Features and Get Adversary Guess
            adversary_guess_flat = model_adversary(latent_z)

            # Check on performance
            batch_y_flat = batch_prv[:,-1,:].view(-1, batch_prv.shape[-1]) # Grab only last in sequeence
            adv_train_loss = F.mse_loss(adversary_guess_flat, batch_y_flat)
            model_adversary.zero_grad()
            adv_train_loss.backward()
            opt_adversary.step()

            ########################################
            # 2. Calculate the Recon Loss
            ########################################
            # Get the latent features and sanitized data
            latent_z, sanitized_data, kl_divergence = model_vae(batch_all)
            pub_prediction = batch_pub[:,-1,:]

            # Take Latent Features and Get Adversary Guess
            adversary_guess_flat = model_adversary(latent_z)
            # Check on performance
            batch_y_flat = batch_prv[:,-1,:].view(-1, batch_prv.shape[-1])
            pub_recon_loss = F.mse_loss(sanitized_data[:, pub_columns], pub_prediction)
            adv_loss = F.mse_loss(adversary_guess_flat, batch_y_flat)
            final_loss_scalar = pub_recon_loss - 4.0 * adv_loss + kl_dig_hypr * kl_divergence.mean()

            recon_losses.append(pub_recon_loss.mean().item())
            adv_losses.append(adv_loss.mean().item())

            model_vae.zero_grad()
            final_loss_scalar.backward()
            opt_vae.step()
            # logger.info(f"Epoch {e} Batch {b} Recon Loss is {recon_loss} and Adversary Loss is {adv_loss}")

            if wandb_on:
                wandb.log({
                    "adv_train_loss": adv_train_loss.item(),
                    "pub_recon_loss": pub_recon_loss.item(),
                    "final_loss_scalar": final_loss_scalar.item(),
                })

            if batch_no % 16 == 0:
                # TODO: Finish the validation implementaion with correlation
                # - Log the validation metrics here
                model_vae.eval()
                model_adversary.eval()
                with torch.no_grad():
                    validation_metrics = (
                        calculate_validation_metrics(
                            all_valid_seqs,
                            pub_columns,
                            prv_columns,
                            model_vae,
                            model_adversary,
                        )
                    )
                model_vae.train()
                model_adversary.train()

                # Report to wandb
                if wandb_on:
                    wandb.log(validation_metrics)

    return model_vae, model_adversary, recon_losses, adv_losses


# TODO: We need to implement federated learning in this particular part of the expression
def federated():
    # We also need a federated aspect to all this. And its getting close to being time to implementing this
    raise NotImplementedError


def triv_calculate_validation_metrics(
    all_features: torch.Tensor,
    pub_features_idxs: List[int],
    prv_features_idxs: List[int],
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


def baseline_pca_decorrelation(
    ds_train: OrderedDict[str, np.ndarray],
    ds_val: OrderedDict[str, np.ndarray],
    prv_features_idxs: List[int],
    batch_size: int,
    correlation_threshold: float,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Tuple[nn.Module, np.ndarray]:
    pca = PCA()

    first_key = list(ds_train.keys())[0]
    num_features = ds_train[first_key].shape[-1]
    pub_features_idxs = list(set(range(num_features)) - set(prv_features_idxs))
    sequence_length = ds_train[first_key].shape[1]

    # We need to concatenate the features as usual 
    all_train_seqs = np.concatenate([ seqs for _, seqs in ds_train.items()], axis=0)
    all_valid_seqs = np.concatenate([ seqs for _, seqs in ds_val.items()], axis=0)
    # Shuffle, Batch, Torch Coversion, Feature Separation
    
    all_train_seqs = torch.from_numpy(all_train_seqs).to(torch.float32).to(device)
    all_valid_seqs = torch.from_numpy(all_valid_seqs).to(torch.float32).to(device)
    train_pub = all_train_seqs[:,:,pub_features_idxs]
    train_prv = all_train_seqs[:,:,prv_features_idxs].squeeze()

    train_pub_flat = train_pub.view(-1, train_pub.shape[-1]).cpu().numpy()
    train_pub_centered = train_pub_flat - np.mean(train_pub_flat, axis=0)
    train_prv_flat = train_prv.view(-1).cpu().numpy()

    #  Now we can start fitting the pca 
    pca_transform = pca.fit(train_pub_flat)
    # Shape is (num_components, num_features), fyi for indexing purposes
    principal_components = pca_transform.components_
    # Takes in (num_componets, num_features) and returns (num_components, num_samples)
    pub_pc_scores = train_pub_flat.dot(principal_components.T)

    retained_components = []
    for i in range(principal_components.shape[0]):
        pc_i_scores = pub_pc_scores[:,i]
        corr_i = np.corrcoef(pc_i_scores, train_prv_flat)[0,1]
        if abs(corr_i) <= correlation_threshold:
            retained_components.append(principal_components[i])

    retained_components = np.array(retained_components)
    # Project the data onto the retained components
    sanitized_projections_for_training = train_pub_centered.dot(retained_components.T)
    sanitized_feature_nums = retained_components.shape[0]
    sanitized_projections_for_training = sanitized_projections_for_training.reshape(-1, sequence_length, sanitized_feature_nums)
    sanitized_projections_for_training = torch.from_numpy(sanitized_projections_for_training).to(torch.float32).to(device)
    num_batches = ceil(sanitized_projections_for_training.shape[0] / batch_size)

    # We need to restructure the data.
    logger.info("Restructuring the data")

    criterion = nn.MSELoss()
    adversary = TrivialTemporalAdversary(
        num_pub_features=sanitized_feature_nums,
        num_prv_features=len(prv_features_idxs),
        dnn_hidden_size=31,
        rnn_hidden_size=30,
    ).to(device)
    opt_adversary = torch.optim.Adam(adversary.parameters(), lr=lr) # type: ignore

    ########################################
    # Now we start the training on the components.
    ########################################
    reconstruction_losses = []
    for e in tqdm(range(epochs), desc="Epochs"):
        logger.info(f"Epoch {e} of {epochs}")
        for batch_no in tqdm(range(num_batches), desc="Batches"):
            # Now Get the new VAE generations
            batch_pub = sanitized_projections_for_training[batch_no * batch_size : (batch_no + 1) * batch_size]
            batch_prv = train_prv[batch_no * batch_size : (batch_no + 1) * batch_size]

            adversary_guess = adversary(batch_pub).squeeze()

            # Calculate the loss
            loss = criterion(adversary_guess, batch_prv[:,-1])
            adversary.zero_grad()
            loss.backward()
            opt_adversary.step()
            reconstruction_losses.append(loss.item())

            if wandb_on:
                wandb.log({
                    "adv_train_loss": loss.item(),
                })
            print("Here")

    # Plot reconstruction losses
    plt.plot(reconstruction_losses)
    plt.savefig(f"./figures/pca_recon_losses.png")
    plt.close()

    return adversary, retained_components




def baseline_trivial_correlation(
    ds_train: OrderedDict[str, np.ndarray],
    ds_val: OrderedDict[str, np.ndarray],
    all_columns: List[str],
    prv_features_idxs: List[int],
    batch_size,
    epochs: int,
    lr: float,
    ds_test: np.ndarray,
    device: torch.device,
):
    """
    Will try to remove a column and simply try to predict it out of the other ones. 
    """
    pub_features_idxs  = list(set(range(len(all_columns))) - set(prv_features_idxs))

    all_train_seqs = np.concatenate([ seqs for _, seqs in ds_train.items()], axis=0)
    all_valid_seqs = np.concatenate([ seqs for _, seqs in ds_val.items()], axis=0)
    # Shuffle, Batch, Torch Coversion, Feature Separation
    np.random.shuffle(all_train_seqs)
    np.random.shuffle(all_valid_seqs)
    batch_amnt  = all_train_seqs.shape[0] // batch_size
    all_train_seqs = torch.from_numpy(all_train_seqs).to(torch.float32).to(device)
    all_valid_seqs = torch.from_numpy(all_valid_seqs).to(torch.float32).to(device)
    train_pub = all_train_seqs[:,:,pub_features_idxs]
    train_prv = all_train_seqs[:,:,prv_features_idxs]

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
        logger.info(f"Epoch {e} of {epochs}")
        for batch_no in tqdm(range(num_batches), desc="Batches"):
            batch_all = all_train_seqs[batch_no * batch_size : (batch_no + 1) * batch_size]
            batch_pub = train_pub[batch_no * batch_size : (batch_no + 1) * batch_size]
            batch_prv = train_prv[batch_no * batch_size : (batch_no + 1) * batch_size]
            if batch_pub.shape[0] != batch_size:
                continue

            logger.info(f"Batch {batch_no} out of {num_batches} with shape {batch_pub.shape}")

            ########################################
            # 1. Get the adversary to guess the sensitive column
            ########################################
            # Get the latent features and sanitized data
            adversary_guess_flat = trivial_adversary(batch_pub)

            # Check on performance
            batch_y_flat = batch_prv[:,-1,:].view(-1, batch_prv.shape[-1]) # Grab only last in sequeence
            adv_train_loss = F.mse_loss(adversary_guess_flat, batch_y_flat)
            trivial_adversary.zero_grad()
            adv_train_loss.backward()
            opt_adversary.step()

            adv_losses.append(adv_train_loss.item())


            if wandb_on:
                wandb.log({
                    "triv_adv_train_loss": adv_train_loss.item(),
                })

            if batch_no % 16 == 0:
                # TODO: Finish the validation implementaion with correlation
                # - Log the validation metrics here
                trivial_adversary.eval()
                with torch.no_grad():
                    validation_metrics = (
                        triv_calculate_validation_metrics(
                            all_valid_seqs,
                            pub_features_idxs,
                            prv_features_idxs,
                            trivial_adversary,
                        )
                    )
                trivial_adversary.train()

                # Report to wandb
                if wandb_on:
                    wandb.log(validation_metrics)
    return trivial_adversary, adv_losses


def baseline_kosambi_karhunen_loeve(test_runs: OrderedDict[str, np.ndarray], pretrained_adversary: nn.Module):
    """
    Kosambi Karhunen Loeve baseline
    """
    pass


def main():
    args = argsies()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(args.seed)
    logger = create_logger("main_training")

    if args.wandb:
        wandb_on = True
        wandb.init(project="private_control", config=args)

    if args.debug:
        logger.info("\033[1;33m Waiting for debugger to attach...\033[0m")
        debugpy.listen(("0.0.0.0", args.debug_port))
        debugpy.wait_for_client()

    # TODO: Make it  so that generate_dataset checks if params are the same
    columns, runs_dict, debug_file = load_defacto_data(args.defacto_data_raw_path)
    num_columns = len(columns)
    num_private_cols = len(args.cols_to_hide)

    # Separate them into their splits (and also interpolate)
    train_seqs, val_seqs, test_file = split_defacto_runs(
        runs_dict,
        args.splits["train_split"],
        args.splits["val_split"],
        args.episode_length,
        True, # Scale
    )

    logger.info(f"Using device is {device}")

    # Get Informaiton for the VAE
    # vae_input = len(columns) - len(args.cols_to_hide)  # For when we want to send only the public ones
    vae_input_size = len(columns) # I think sending all of them is better
    # pub_dimensions 
    # TODO: Get the model going
    model_vae = SequenceToScalarVAE(
        # inout_size for model is output_dim for data
        input_size=vae_input_size,
        latent_size=args.vae_latent_size,
        hidden_size=args.vae_hidden_size,
        num_features_to_guess=1,
        rnn_num_layers=args.rnn_num_layers,
        rnn_hidden_size=args.rnn_hidden_size,
    ).to(device)

    model_adversary = Adversary(
        input_size=args.vae_latent_size,
        hidden_size=args.vae_hidden_size,
        num_classes=num_private_cols,
    ).to(device)

    logger.debug(f"Columns are {columns}")
    logger.debug(f"Runs dict is {runs_dict}")

    # TOREM: Move this lower down for when we are done with it. 
    pca_model_adversary, retained_components = baseline_pca_decorrelation(
        train_seqs,
        val_seqs,
        args.cols_to_hide,
        args.batch_size,
        args.correlation_threshold,
        args.epochs,
        args.lr,
        device,
    )
    # Now we run the test on this 
    pca_test_entire_file(
        test_file,
        args.cols_to_hide,
        pca_model_adversary,
        retained_components,
        args.episode_length,
        args.padding_value,
        logger,
        args.batch_size,
        wandb_on=args.wandb
    )
    exit() # TOREM:

    ########################################
    # Training VAE and Adversary
    ########################################
    logger.info("Starting the VAE Training")
    model_vae, model_adversary, recon_losses, adv_losses = train_v1(
        args.batch_size,
        args.cols_to_hide,
        columns,
        device,
        train_seqs,
        val_seqs,
        args.epochs,
        model_vae,
        model_adversary,
        args.lr,
        args.kl_dig_hypr,
    )

    ########################################
    # Evaluation
    ########################################
    plot_training_losses(recon_losses, adv_losses, f"./figures/new_data_vae/recon-adv_losses.png")

    # TODO: Move this to a test 
    metrics = test_entire_file(
        test_file,
        args.cols_to_hide,
        model_vae,
        model_adversary,
        args.episode_length,
        args.padding_value,
        args.batch_size,
        wandb_on=args.wandb
    )
    logger.info(f"Validation Metrics are {metrics}")

    # Benchmarks: 

    ########################################
    # Training Trivial Adversary
    ########################################
    trivial_adverary, adv_losses = baseline_trivial_correlation(
        train_seqs,
        val_seqs,
        columns,
        args.cols_to_hide,
        args.batch_size, args.epochs,
        args.lr,
        test_file,
        device,
    )
    triv_test_entire_file(
        test_file,
        args.cols_to_hide,
        trivial_adverary,
        args.episode_length,
        args.padding_value,
        args.batch_size,
        wandb_on=args.wandb
    )


    # kd_transform_baseline(test_runs, model_adversary)

    # 🚩 development so far🚩
    exit()

if __name__ == "__main__":
    logger = create_logger("main_vae")
    main()
