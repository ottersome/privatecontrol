import os
from typing import Optional, Sequence, Dict, Tuple
from math import ceil
import logging

from matplotlib.axes import Axes
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb

from conrecon.data.dataset_generation import (
    batch_generation_randUni,
    collect_n_sequential_batches,
    spot_backhistory,
)
from conrecon.utils.graphing import plot_comp, plot_given, plot_signal_reconstructions, plot_single_signal_reconstruction


def triv_test_entire_file(
    test_file: np.ndarray,
    idxs_colsToGuess: Sequence[int],
    model_adversary: nn.Module,
    sequence_length: int,
    padding_value: Optional[int],
    batch_size: int = 16,
    wandb_on: bool = False,
) -> Dict[str, float]:
    """
    Will run a validation iteration for a model
    Args:
        - validation_file: Validation data (file_length, num_features)
        - model: Model to run the validation on
        - col_to_predict: (0-index) which column we would like to predict
    """
    metrics = {
        "recon_loss": [],
        "adv_loss": [],
    }
    amnt_columns = test_file.shape[-1]
    pub_features_idxs = list(set(range(amnt_columns)) - set(idxs_colsToGuess))
    print(f"lengtth of pub_features_idxs is {len(pub_features_idxs)}")

    model_adversary.eval()
    device = next(model_adversary.parameters()).device
    val_x = torch.from_numpy(test_file).to(torch.float32).to(device)

    # Generate the reconstruction
    private_columns = list(idxs_colsToGuess)
    num_batches = ceil((len(val_x) - sequence_length) / batch_size)

    batch_guesses = []
    for batch_no in range(num_batches):
        ########################################
        # Sanitize the data
        ########################################
        start_idx = batch_no * batch_size + sequence_length
        end_idx = min((batch_no + 1) * batch_size + sequence_length, val_x.shape[0])
        backhistory = collect_n_sequential_batches(
            val_x, start_idx, end_idx, sequence_length, padding_value
        )
        backhistory_pub = backhistory[:, :, pub_features_idxs]
        # print(f"backhistory_pub shape is {backhistory_pub.shape}")

        # TODO: Incorporate Adversary Guess
        adversary_guess = model_adversary(backhistory_pub)
        batch_guesses.append(adversary_guess)

    seq_guesses = torch.cat(batch_guesses, dim=0)

    ########################################
    # Chart for Adversary
    ########################################
    adv_to_show = seq_guesses[:, :].cpu().detach().numpy()
    adv_truth = test_file[:, private_columns]
    print(f"Adv truth shape is {adv_truth.shape}")
    print(f"Adv to show shape is {adv_to_show.shape}")

    plot_single_signal_reconstruction(
        adv_truth.squeeze(),
        adv_to_show.squeeze(),
        "./figures/vae_non_sanitized.png",
        "Reconstruction Vs Truth of $f_s$: Non-Sanitized",
    )

    # fig = plt.figure(figsize=(16, 10))
    # plt.plot(adv_to_show.squeeze().detach().cpu().numpy(), label="Adversary")
    # plt.title("Adversary")
    # plt.legend()
    # plt.plot(adv_truth.squeeze(), label="Truth")
    # plt.title("Truth")
    # plt.legend()
    #
    # plt.savefig(f"figures/triv_adversary.png")
    # plt.close()

    # Pass reconstruction and adversary to wandb
    if wandb_on:
        wandb.log({"adversary": adv_to_show.squeeze().detach().cpu().numpy()})

    model_adversary.train()

    return {k: np.mean(v).item() for k, v in metrics.items()}


def pca_test_entire_file(
    test_file: np.ndarray,
    prv_features_idxs: Sequence[int],
    model_recon: nn.Module,
    model_adversary: nn.Module,
    principal_components: torch.Tensor,
    train_mean: np.ndarray,
    sequence_length: int,
    padding_value: Optional[int],
    logger: logging.Logger,
    batch_size: int = 16,
    wandb_on: bool = False,
) -> tuple[Dict[str, float], float, float]:
    """
    Will run a the test iteration on model with adverasry
    Args:
        - test_file: Validation data (file_length, num_features)
        - prv_features_idxs: (0-index) which column we would like to predict
        - model_recon: Model that will be used to reconstruct public data
        - model_adversary: Model that will be used to adversariall reconstruct sensitive data
        - principal_components: Principal Components
        - sequence_length: Sequence length
        - padding_value: Padding value
        - batch_size: Batch size
        - wandb_on: Wandb on
    returns:
        - metrics: Dict of metrics:02d
        - utility: Utility of the model after removing the most salient feature
        - privacy: Privacy of the model after removing the most salient feature
    """
    metrics = {
        "recon_loss": [],
        "adv_loss": [],
    }
    model_recon.eval()
    model_adversary.eval()

    device = next(model_adversary.parameters()).device
    test_tensor = torch.from_numpy(test_file).to(torch.float32).to(device)
    train_mean_tensor = torch.from_numpy(train_mean).to(torch.float32).to(device)

    num_batches = ceil((len(test_tensor) - sequence_length) / batch_size)

    batch_adv_infs = []
    batch_recon_infs = []
    ######################################## Adversarial Reconstructions
    ########################################
    for batch_no in range(num_batches):
        ########################################
        # Sanitize the data
        ########################################
        start_idx = batch_no * batch_size + sequence_length
        end_idx = min(
            (batch_no + 1) * batch_size + sequence_length, test_tensor.shape[0]
        )
        backhistory = collect_n_sequential_batches(
            test_tensor, start_idx, end_idx, sequence_length, padding_value
        )
        backhistory_centered = backhistory - train_mean_tensor
        projected_backhistory = torch.matmul(backhistory_centered, principal_components.T)
        # pca_projected_ds = train_all_centered.dot(principal_components.T)

        # TODO: Incorporate Adversary Guess
        with torch.no_grad():
            recon_inference = model_recon(projected_backhistory)
            adversary_guess = model_adversary(projected_backhistory)

        batch_recon_infs.append(recon_inference)
        batch_adv_infs.append(adversary_guess)

    adv_guesses = torch.cat(batch_adv_infs, dim=0).cpu()
    seq_reconstructions = torch.cat(batch_recon_infs, dim=0)
    ########################################
    # Chart for Reconstruction
    ########################################
    num_principal_components = principal_components.shape[0]
    logger.info(f"Plotting reconstruction with {num_principal_components} components")
    pub_features_idxs = list(set(range(test_file.shape[-1])) - set(prv_features_idxs))
    print(pub_features_idxs)
    os.makedirs("./figures/method2", exist_ok=True)
    plot_comp(
        test_file,
        seq_reconstructions.cpu(),
        pub_features_idxs,
        f"figures/method2/pca_reconstruction_{num_principal_components:02d}_componentsRem.png",
    )

    ########################################
    # Compute metrics to plot later
    ########################################
    utility = -1 * torch.mean((seq_reconstructions.cpu() - test_file[sequence_length:, pub_features_idxs]) ** 2).item()
    privacy = torch.mean((adv_guesses.cpu() - test_file[sequence_length:, prv_features_idxs]) ** 2).item()

    ########################################
    # Chart for Adversary
    ########################################
    adv_to_show = adv_guesses[:, :]
    adv_truth = test_file[sequence_length:, prv_features_idxs]

    plot_signal_reconstructions(
        test_file[sequence_length:, prv_features_idxs],
        adv_guesses,
        f"figures/method2/pca_adversary_{num_principal_components:02d}_componentsRem",
    )

    # Pass reconstruction and adversary to wandb
    if wandb_on:
        wandb.log({"adversary": adv_to_show.squeeze().detach().cpu().numpy()}) # type: ignore

    model_adversary.train()

    return {k: np.mean(v).item() for k, v in metrics.items()}, utility, privacy

def nonAdvVAE_test_file(
    test_file: np.ndarray,
    idxs_ignoredCols: Sequence[int],
    model_vae: nn.Module,
    sequence_length: int,
    dump_path_data_results: str,
    dump_path_figures: str,
    batch_size: int,
) -> Dict[str, float]:
    """
    Will run a visualized test iteration for the passed model
    Args:
        - validation_file: Validation data (file_length, num_features)
        - model: Model to run the validation on
        - col_to_predict: (0-index) which column we would like to predict
    """
    metrics = {}
    all_dirs = [ dump_path_data_results, dump_path_figures ]
    for dir in all_dirs:
        dirname = os.path.dirname(dir)
        os.makedirs(dirname, exist_ok=True)

    model_vae.eval()
    model_device = next(model_vae.parameters()).device
    test_x = torch.from_numpy(test_file).to(torch.float32).to(model_device)

    # Generate the reconstruction
    num_batches = ceil(
        (len(test_x) - sequence_length) / batch_size
    )  ## Subtract sequence_length to avoid padding
    evaluation_initial_offset = sequence_length - 1

    batch_sanitized = []
    latent_zs = []
    with torch.no_grad():
        for batch_no in range(num_batches):
            ########################################
            # Sanitize the data
            ########################################
            start_idx = (
                batch_no * batch_size + evaluation_initial_offset
            )  #  Sequence length to avoid padding
            end_idx = min((batch_no + 1) * batch_size + evaluation_initial_offset, test_x.shape[0])
            backhistory = collect_n_sequential_batches(
                test_x, start_idx, end_idx, sequence_length
            )
            latent_z, sanitized_data, kl_divergence = model_vae(backhistory)

            # TODO: Incorporate Adversary Guess
            batch_sanitized.append(sanitized_data)
            latent_zs.append(latent_z)

    ########################################
    # Dump data to play with it later.
    ########################################
    seq_sanitized = torch.cat(batch_sanitized, dim=0)
    seq_latent_zs = torch.cat(latent_zs, dim=0)

    tosave_val_x = test_x.cpu().numpy()

    tosave_sanitized = seq_sanitized.cpu().numpy()
    tosave_latent_zs = seq_latent_zs.cpu().numpy()
    os.makedirs("./results/", exist_ok=True)
    np.save(os.path.join(dump_path_data_results,f"val_x_{idxs_ignoredCols}.npy"), tosave_val_x)
    np.save(os.path.join(dump_path_data_results,f"sanitized_x_{idxs_ignoredCols}.npy"), tosave_sanitized)
    np.save(os.path.join(dump_path_data_results,f"latent_z_{idxs_ignoredCols}.npy"), tosave_latent_zs)

    ########################################
    # Chart For Reconstruction
    ########################################

    recon_to_show = seq_sanitized
    truth_to_compare = torch.from_numpy(test_file[evaluation_initial_offset:]).to(model_device)
    # pub_features_idxs = list(set(range(test_file.shape[-1])) - set(idxs_colsToGuess))
    path_to_save_fig = os.path.join(dump_path_figures, "vae_pure_recon.png")
    assert (
        truth_to_compare.shape[0] == recon_to_show.shape[0]
    ), f"Shape mismatch: truth_to_compare.shape is {truth_to_compare.shape} and recon_to_show.shape is {recon_to_show.shape}"

    public_cols = list(set(range(truth_to_compare.shape[-1])) - set(idxs_ignoredCols))
    metrics["recon_loss"] = F.mse_loss(truth_to_compare[:, public_cols], recon_to_show).item()


    plot_comp(
        truth_to_compare.cpu(),
        recon_to_show.cpu(),
        public_cols,
        path_to_save_fig
    )

    
    model_vae.train()

    return metrics

def advVae_test_file(
    test_file: np.ndarray,
    idxs_colsToGuess: Sequence[int],
    model_vae: nn.Module,
    model_adversary: nn.Module,
    sequence_length: int,
    padding_value: Optional[int],
    batch_size: int = 16,
    wandb_on: bool = False,
    recon_savefig_loc: str = "./figures/model_vae/vae_reconstruction.png",
    adv_savefig_loc: str = "./figures/model_vae/vae_adversary.png",
) -> Dict[str, float]:
    """
    Will run a validation iteration for a model
    Args:
        - validation_file: Validation data (file_length, num_features)
        - model: Model to run the validation on
        - col_to_predict: (0-index) which column we would like to predict
    """
    metrics = {
        "recon_loss": [],
        "adv_loss": [],
    }
    recon_base_dir = os.path.basename(recon_savefig_loc)
    adv_base_dir = os.path.basename(adv_savefig_loc)
    os.makedirs("./results/model_vae/", exist_ok=True)
    os.makedirs("./figures/model_vae/", exist_ok=True)

    model_vae.eval()
    model_adversary.eval()
    device = next(model_vae.parameters()).device

    model_device = next(model_vae.parameters()).device
    test_x = torch.from_numpy(test_file).to(torch.float32).to(model_device)

    # Generate the reconstruction
    public_columns = list(set(range(test_x.shape[-1])) - set(idxs_colsToGuess))
    private_columns = list(idxs_colsToGuess)
    num_batches = ceil(
        (len(test_x) - sequence_length) / batch_size
    )  ## Subtract sequence_length to avoid padding
    evaluation_initial_offset = sequence_length - 1

    batch_guesses = []
    batch_sanitized = []
    latent_zs = []
    with torch.no_grad():
        for batch_no in range(num_batches):
            ########################################
            # Sanitize the data
            ########################################
            start_idx = (
                # This TESTFILE looks super sus. Why would we multiply batch_no y batch_size
                batch_no * batch_size + evaluation_initial_offset
            )  #  Sequence length to avoid padding
            end_idx = min((batch_no + 1) * batch_size + evaluation_initial_offset, test_x.shape[0] - 1)
            backhistory = collect_n_sequential_batches(
                test_x, start_idx, end_idx, sequence_length, padding_value
            )
            latent_z, sanitized_data, kl_divergence = model_vae(backhistory)

            # TODO: Incorporate Adversary Guess
            adversary_guess = model_adversary(latent_z)
            batch_guesses.append(adversary_guess)
            batch_sanitized.append(sanitized_data)
            latent_zs.append(latent_z)

    ########################################
    # Dump data to play with it later.
    ########################################
    seq_adv_guesses = torch.cat(batch_guesses, dim=0)
    seq_sanitized = torch.cat(batch_sanitized, dim=0)
    seq_latent_zs = torch.cat(latent_zs, dim=0)

    tosave_val_x = test_x.cpu().numpy()

    tosave_advGuesses_y = seq_adv_guesses.cpu().numpy()
    tosave_sanitized = seq_sanitized.cpu().numpy()
    tosave_latent_zs = seq_latent_zs.cpu().numpy()
    os.makedirs("./results/", exist_ok=True)
    np.save(f"./results/val_x_{idxs_colsToGuess}.npy", tosave_val_x)
    np.save(f"./results/adv_guesses_y_{idxs_colsToGuess}.npy", tosave_advGuesses_y)
    np.save(f"./results/sanitized_x_{idxs_colsToGuess}.npy", tosave_sanitized)
    np.save(f"./results/latent_z_{idxs_colsToGuess}.npy", tosave_latent_zs)

    ########################################
    # Chart For Reconstruction
    ########################################

    recon_to_show = seq_sanitized
    truth_to_compare = test_file[evaluation_initial_offset:]
    pub_features_idxs = list(set(range(test_file.shape[-1])) - set(idxs_colsToGuess))
    assert (
        truth_to_compare.shape == recon_to_show.shape
    ), f"Shape mismatch: truth_to_compare.shape is {truth_to_compare.shape} and recon_to_show.shape is {recon_to_show.shape}"
    plot_comp(
        truth_to_compare,
        recon_to_show.cpu(),
        pub_features_idxs,
        f"figures/method_vae/vae_reconstruction.png",
    )


    adv_truth = test_file[:, private_columns]
    adv_to_show = seq_adv_guesses[:, :].cpu().detach().numpy()
    plot_signal_reconstructions(
        adv_truth,
        adv_to_show,
        f"figures/method_vae/vae_adversary",
    )


    # # Lets now save the figure
    # permd_sanitized_idxs = torch.randperm(len(public_columns))[:8]
    # permd_original_idxs = torch.tensor(
    #     [public_columns[idx] for idx in permd_sanitized_idxs]
    # ).to(torch.long)
    #
    # recon_to_show = seq_sanitized[:, permd_sanitized_idxs]
    # truth_to_compare = test_file[:, permd_original_idxs]
    # fig, axs = plt.subplots(4, 2, figsize=(32, 20))
    # for i in range(recon_to_show.shape[1]):
    #     mod = i % 4
    #     idx = i // 4
    #     axs[mod, idx].plot(
    #         recon_to_show[:, i].squeeze().detach().cpu().numpy(), label="Reconstruction"
    #     )
    #     axs[mod, idx].set_title(
    #         f"Reconstruction Vs Truth of $f_{permd_original_idxs[i]}$"
    #     )
    #     axs[mod, idx].legend()
    #     axs[mod, idx].plot(truth_to_compare[:, i].squeeze(), label="Truth")
    #     axs[mod, idx].legend()
    #     if wandb_on:
    #         wandb.log({f"Reconstruction (Col {i})": recon_to_show[:, i].squeeze().detach().cpu().numpy() })
    #         wandb.log({f"Truth (Col {i})": truth_to_compare[:, i].squeeze().detach().cpu().numpy() }
    #         )
    # plt.savefig(recon_savefig_loc)
    # plt.close()
    
    ########################################
    # Chart for Adversary
    ########################################
    # adv_to_show = seq_adv_guesses[:, :]
    # adv_truth = test_file[:, private_columns]
    #
    # fig = plt.figure(figsize=(16, 10))
    # plt.plot(adv_to_show.squeeze().detach().cpu().numpy(), label="Adversary")
    # plt.title("Adversary")
    # plt.legend()
    # plt.plot(adv_truth.squeeze(), label="Truth")
    # plt.title("Truth")
    # plt.legend()
    #
    # plt.savefig(adv_savefig_loc)
    # plt.close()
    #
    # # Pass reconstruction and adversary to wandb
    # if wandb_on:
    #     wandb.log({"reconstruction": recon_to_show.squeeze().detach().cpu().numpy()})
    #     wandb.log({"adversary": adv_to_show.squeeze().detach().cpu().numpy()})

    model_vae.train()
    model_adversary.train()

    return {k: np.mean(v).item() for k, v in metrics.items()}


def get_tradeoff_metrics(
    test_file: np.ndarray,
    idxs_colsToGuess: Sequence[int],
    model_vae: nn.Module,
    model_adversary: nn.Module,
    sequence_length: int,
    padding_value: Optional[int],
    batch_size: int = 16,
) -> Tuple[float, float]:
    """
    Will run a validation iteration for a model
    Args:
        - validation_file: Validation data (file_length, num_features)
        - model: Model to run the validation on
        - col_to_predict: (0-index) which column we would like to predict
    """
    ########################################
    # Setup some simple stuff here
    ########################################
    model_vae.eval()
    model_adversary.eval()
    device = next(model_vae.parameters()).device
    test_x = torch.from_numpy(test_file).to(torch.float32).to(device)
    public_columns = list(set(range(test_x.shape[-1])) - set(idxs_colsToGuess))
    private_columns = list(idxs_colsToGuess)
    test_pub = test_x[sequence_length:, public_columns]
    test_prv = test_x[sequence_length:, private_columns]
    num_batches = ceil(
        (len(test_x) - sequence_length) / batch_size
    )  ## Subtract sequence_length to avoid padding

    ########################################
    # Go through all the public data first
    ########################################
    batch_guesses = []
    batch_reconstructions = []
    for batch_no in range(num_batches):
        ########################################
        # Sanitize the data
        ########################################
        start_idx = (
            batch_no * batch_size + sequence_length
        )  #  Sequence length to avoid padding
        end_idx = min((batch_no + 1) * batch_size + sequence_length, test_x.shape[0])
        backhistory = collect_n_sequential_batches(
            test_x, start_idx, end_idx, sequence_length, padding_value
        )
        with torch.no_grad():
            latent_z, sanitized_data, kl_divergence = model_vae(backhistory)
            adversary_guess = model_adversary(latent_z)
        batch_guesses.append(adversary_guess)
        batch_reconstructions.append(sanitized_data)

    seq_guesses = torch.cat(batch_guesses, dim=0)
    seq_reconstructions = torch.cat(batch_reconstructions, dim=0)

    # Now we create our average metrics
    recon_mse = torch.mean((seq_reconstructions - test_pub) ** 2)
    adv_mse = torch.mean((seq_guesses - test_prv) ** 2)

    return adv_mse.item(), -1 * recon_mse.item()


def test_pca_M_decorrelation(
    recovered_private_guess: np.ndarray,
    recovered_public_guess: np.ndarray,
    test_pub: np.ndarray,
    test_prv: np.ndarray,
    num_features_being_kept: int,
):
    # os.makedirs("./figures/pca_triv/", exist_ok=True)
    # # DEBUG: FOr now we plot it like this
    # # Plot the private reconstruction
    # plt.figure(figsize=(16,10))
    # plt.plot(recovered_private_guess.squeeze(), label="Reconstruction")
    # plt.plot(test_prv_flattend, label="Truth")
    # plt.title("PCA Priv Reconstruction vs Truth")
    # plt.legend()
    # plt.savefig(f"./figures/pca_triv/heatmap_mui_priv_reconstruction_{num_features_to_keep}.png")
    # plt.close()
    os.makedirs("./figures/method1", exist_ok=True)

    # # Now the same but for public
    plot_given(
        test_pub,
        recovered_public_guess,
        "Public Feature Inference",
        "True Public Features",
        "Time",
        "Magnitude",
        "Feature {} True vs Inference",
        "Feature Reconstruction Evaluation",
        f"./figures/method1/heatmap_mui_pub_reconstruction_{num_features_being_kept:02d}.png",
    )

    plot_signal_reconstructions(
        test_prv,
        recovered_private_guess,
        f"./figures/method1/heatmap_mui_prv_reconstruction_{num_features_being_kept:02d}.png",
    )
