import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import torch
from rich import traceback
from rich.console import Console
from rich.live import Live
# from sktime.libs.pykalman import KalmanFilter
from torch import nn, tensor
from torch.nn import functional as F
from tqdm import tqdm

from conrecon.data.dataset_generation import TrainingMetaData, generate_dataset
from conrecon.dplearning.vae import FlexibleVAE
from conrecon.kalman.mo_core import Filter
from conrecon.plotting import TrainLayout, plot_functions, plot_functions_2by1
from conrecon.ss_generation import hand_design_matrices
from conrecon.utils.common import create_logger

traceback.install()

console = Console()

def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # State stuff here
    ap.add_argument(
        "-e", "--epochs", default=3, help="How many epochs to train for", type=int
    )
    ap.add_argument(
        "--num_layers", default=1, help="How many epochs to train for", type=int
    )
    ap.add_argument(
        "--eval_interval", default=100, help="How many epochs to train for", type=int
    )
    ap.add_argument(
        "-n", "--num_samples", default=4, type=int, help="How many Samples to Evaluate"
    )
    ap.add_argument(
        "--eval_size", default=4, help="How many systems to generate", type=int
    )
    # Control stuff here
    ap.add_argument(
        "-t", "--time_steps", default=12, help="How many systems to generate", type=int
    )
    ap.add_argument(
        "-s", "--state_dim", default=3, help="Dimensionality of the state.", type=int
    )
    ap.add_argument(
        "-i", "--input_dim", default=3, help="Dimensionality of the input.", type=int
    )
    ap.add_argument(
        "-o", "--output_dim", default=2, help="Dimensionality of the output.", type=int
    )
    ap.add_argument("--defacto_data_raw", default="./data/defacto_data.csv", type=str)
    ap.add_argument("--ds_cache", default=".cache/pykalpkg_ds.csv", type=str)
    ap.add_argument("--vae_ds_cache", default=".cache/pykalpkg_vaeds.csv", type=str)
    ap.add_argument(
        "--saveplot_dest",
        default="./figures/pykalman_transformer/",
        help="Where to save the outputs",
    )
    ap.add_argument("--ds_size", default=10000, type=int)
    ap.add_argument("--no-autoindent")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--lr", default=0.001, type=float)
    ap.add_argument("--first_n_states", default=7, type=int)

    args = ap.parse_args()

    if not os.path.exists(args.saveplot_dest):
        os.makedirs(args.saveplot_dest)
    if not os.path.exists(".cache/"):
        os.makedirs(".cache/")
    return args
    # Sanity check



#💫 Main function of interest
def trainVAE_wprivacy(
    training_metadata: TrainingMetaData,
    learning_data: Tuple[np.ndarray, np.ndarray],
    epochs: int,
    plot_dest: str,
    isdim_priv : List[bool],
    batch_size = 64,
    tt_split: float = 0.8,
    vae_latent_size: int = 10,
    vae_hidden_size: int = 128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_valsamps = 2
    ### Learning Objects
    vae = FlexibleVAE(
        # inout_size for model is output_dim for data
        input_size=training_metadata.output_dim,
        latent_size=vae_latent_size,
        hidden_size=vae_hidden_size,
    ).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001) # type: ignore

    ### Filter Data
    A, C, = training_metadata.params.A, training_metadata.params.C
    B = training_metadata.params.B
    assert isinstance(A, np.ndarray), "A should be a numpy array"
    assert isinstance(B, np.ndarray), "B should be a numpy array"
    assert isinstance(C, np.ndarray), "C should be a numpy array"
    # t_data, val_data = learning_data

    # My Filter
    torch_filter = Filter(
        transition_matrix=torch.from_numpy(A).to(torch.float32).to(device),
        observation_matrix=torch.from_numpy(C).to(torch.float32).to(device),
        input_matrix=torch.from_numpy(B).to(torch.float32).to(device),
        batch_size=batch_size,
    ).to(device)
    # Native Filter
    kf = KalmanFilter(transition_matrices=A, observation_matrices=C)


    ### Data Management
    states, outputs = learning_data # x, y
    logger.info(f"Using device is {device}")
    tstates, toutputs = tensor(states), tensor(outputs) # t(x), t(y)
    trn_states, val_states = torch.split(
        tstates,
        [int(len(states) * tt_split), len(states) - int(len(states) * tt_split)],
    )
    trn_outputs, val_outputs = torch.split(
        toutputs,
        [int(len(outputs) * tt_split), len(outputs) - int(len(outputs) * tt_split)],
    )
    t_val_states, t_val_outputs = tensor(val_states).to(device), tensor(val_outputs).to(device)
    ## Prepare Data For Validation
    val_samples = []
    for i in range(num_valsamps):
        state_est,_ = kf.filter(val_outputs[i, :, :].squeeze().detach().cpu().numpy())
        val_samples.append(state_est)
    val_samples = np.stack(val_samples)

    # Make explicit index list for private dims
    num_stateDims = A.shape[1]
    private_dims = []
    public_dims = []
    dim_labels = []
    for i in range(num_stateDims):
        if isdim_priv[i]:
            private_dims.append(i)
            dim_labels.append(f"Private Dim {i}")
        else:
            public_dims.append(i)
            dim_labels.append(f"Public Dim {i}")



    ### Train the VAE
    criterion = nn.MSELoss()  # TODO: Change this to something else
    loss_list = []
    eval_data = []
    batch_count = int(tstates.shape[0] / batch_size)
    train_layout = TrainLayout(epochs, batch_count, loss_list, eval_data)
    with Live(train_layout.layout, console=console, refresh_per_second=1) as live:
        for e in range(epochs):
            epoch_loss = 0
            for b in range(batch_count):
                ## BatchdWise Data
                t_cur_state = trn_states[b * batch_size : (b + 1) * batch_size].to(device)
                cur_output = trn_outputs[b * batch_size : (b + 1) * batch_size].to(
                    device
                )
                if t_cur_state.shape[0] != batch_size:
                    continue; # We are nat dealing with smaller sizes.
                ## Change Data to HideStuff
                masked_output = vae(cur_output)
                state_estimates_w_vae = []
                state_estimates_wo_vae = []

                logger.debug(f"Tell em about the shape of the output {cur_output.shape} as well as its type {type(cur_output)}")
                logger.debug(f"Shape of masked_output is {masked_output.shape} as well as its type {type(masked_output)}")

                # Go Through Batch
                for i in range(cur_output.shape[0]):
                    logger.debug(f"Going through the batch {i}")
                    # First without VAE
                    (filtered_mean, filtered_covariance) = kf.filter(
                        cur_output[i, :, :].squeeze().detach().cpu().numpy()
                    )
                    (smoothed_mean, smoothed_covariance) = kf.smooth(
                        cur_output[i, :, :].squeeze().detach().cpu().numpy()
                    )
                    state_estimates_wo_vae.append(smoothed_mean)

                state_estimates_wo_vae = torch.from_numpy(np.array(state_estimates_wo_vae)).to(device)
                logger.debug(f"Check if tensor requires grad {cur_output.requires_grad}")
                cur_output.requires_grad = True
                state_estimates_w_vae = torch_filter(masked_output)
                logger.debug("Done with the batch. Will add more stuff in a minute")

                # CHECK: This will likely need a torch based implementation to 
                # have the computation graph involved
                logger.debug("Before loss estimation")
                similarities = F.mse_loss(state_estimates_w_vae[:,:,public_dims], t_cur_state[:,:,public_dims])
                diff = - F.mse_loss(state_estimates_wo_vae[:,:,private_dims], t_cur_state[:,:,private_dims])
                # similarities = F.mse_loss(state_estimates_w_vae[:,:,1:], t_cur_state[:,:,1:])
                # diff = - F.mse_loss(state_estimates_wo_vae[:,:,0], t_cur_state[:,:,0])
                final_loss = similarities + diff
                fl_mean = final_loss.mean()
                loss_list.append(fl_mean.item())
                cur_loss = final_loss.mean().item()
                
                logger.debug("Before optimizing.")
                optimizer.zero_grad()
                fl_mean.backward()
                optimizer.step()


                # Plot and Save the Reconstruction of Validation Samples
                inputo = vae(t_val_outputs[:batch_size, :, :])
                val_state_vae_est = torch_filter(inputo).squeeze().cpu().detach().numpy()
                func_to_compare = np.stack(
                    [
                        t_val_states[:num_valsamps, :, :].squeeze().cpu().detach().numpy(),
                        val_state_vae_est[:num_valsamps, :, :],
                        val_samples[:num_valsamps, :, :],
                    ]
                ).transpose(1, 0, 2, 3)
                plot_functions(
                    func_to_compare,
                    f"./figures/vae_fixed_recon/plot_vaerecon_eval_{e:02d}_{b:02d}_recon_states.png",
                    function_labels=["True Value","ITL Method","Control Method"],
                    dim_labels=dim_labels[:2],
                    dims_to_show=[0,1]
                )

                ## Now compare the outputs by saving their plot
                func_to_compare = np.stack(
                    [
                        cur_output[:2].squeeze().cpu().detach().numpy(),
                        masked_output[:2].squeeze().cpu().detach().numpy(),
                    ]
                ).transpose(1, 0, 2, 3)
                plot_functions_2by1(
                    func_to_compare,
                    f"./figures/vae_fixed_recon/plot_vaerecon_eval_{e:02d}_{b:02d}_recon_outputs.png",
                    function_labels=["Ground Truth","ITL Method"],
                    dim_to_show=[1,],
                )


                ## TODO: We need some sort of knob here to play with utility vs privacy
                train_layout.update(e, b, cur_loss, None)

            # Normal Reporting
            if (e + 1) % 1 == 0:
                print("Epoch: {}, Loss: {:.5f}".format(e + 1, epoch_loss))

    # We test with MSE for reconstruction for now
def main():
    args = argsies()
    # Get our A, B, C Matrices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = hand_design_matrices()
    np.random.seed(int(time.time()))

    logger.info("Generating dataset")

    # TODO: Make it  so that generate_dataset checks if params are the same
    hidden, outputs, training_data = generate_dataset(
        params,
        args.ds_cache,
        args.state_dim,
        args.input_dim,
        args.output_dim,
        args.time_steps,
        args.ds_size,
    )

    # With the Dataset in Place we Also Generate a Variational Autoencoder
    # vae = train_VAE(outputs) # CHECK: Should we train this first or remove for later
    # 🚩 Development so far🚩

    trainVAE_wprivacy(
        training_data,
        (hidden, outputs),
        args.epochs,
        args.saveplot_dest,
        [True,False,True]
    )

if __name__ == "__main__":
    logger = create_logger("main_vae")
    main()
