import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import torch
from rich import traceback
from rich.console import Console
from rich.live import Live
from sktime.libs.pykalman import KalmanFilter
from torch import nn, tensor
from torch.nn import functional as F
from tqdm import tqdm

from conrecon.data.data_loading import load_defacto_data, split_defacto_runs
from conrecon.dplearning.vae import FlexibleVAE
from conrecon.kalman.mo_core import Filter
from conrecon.plotting import TrainLayout, plot_functions, plot_functions_2by1
from conrecon.ss_generation import hand_design_matrices
from conrecon.utils import create_logger

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
    ap.add_argument("--defacto_data_raw_path", default="./data/defacto_data.csv", type=str, help="Where to load the data from")
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
# def trainVAE_wprivacy(
#     training_metadata: TrainingMetaData,
#     learning_data: Tuple[np.ndarray, np.ndarray],
#     epochs: int,
#     plot_dest: str,
#     isdim_priv : List[bool],
#     batch_size = 64,
#     tt_split: float = 0.8,
#     vae_latent_size: int = 10,
#     vae_hidden_size: int = 128,
# ):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     num_valsamps = 2
#     ### Learning Objects
#     vae = FlexibleVAE(
#         # inout_size for model is output_dim for data
#         input_size=training_metadata.output_dim,
#         latent_size=vae_latent_size,
#         hidden_size=vae_hidden_size,
#     ).to(device)
#     optimizer = torch.optim.Adam(vae.parameters(), lr=0.001) # type: ignore
#
#     ### Filter Data
#     A, C, = training_metadata.params.A, training_metadata.params.C
#     B = training_metadata.params.B
#     assert isinstance(A, np.ndarray), "A should be a numpy array"
#     assert isinstance(B, np.ndarray), "B should be a numpy array"
#     assert isinstance(C, np.ndarray), "C should be a numpy array"
#     # t_data, val_data = learning_data
#
#     # My Filter
#     torch_filter = Filter(
#         transition_matrix=torch.from_numpy(A).to(torch.float32).to(device),
#         observation_matrix=torch.from_numpy(C).to(torch.float32).to(device),
#         input_matrix=torch.from_numpy(B).to(torch.float32).to(device),
#         batch_size=batch_size,
#     ).to(device)
#     # Native Filter
#     kf = KalmanFilter(transition_matrices=A, observation_matrices=C)
#
#
#     ### Data Management
#     states, outputs = learning_data # x, y
#     logger.info(f"Using device is {device}")
#     tstates, toutputs = tensor(states), tensor(outputs) # t(x), t(y)
#     trn_states, val_states = torch.split(
#         tstates,
#         [int(len(states) * tt_split), len(states) - int(len(states) * tt_split)],
#     )
#     trn_outputs, val_outputs = torch.split(
#         toutputs,
#         [int(len(outputs) * tt_split), len(outputs) - int(len(outputs) * tt_split)],
#     )
#     t_val_states, t_val_outputs = tensor(val_states).to(device), tensor(val_outputs).to(device)
#     ## Prepare Data For Validation
#     val_samples = []
#     for i in range(num_valsamps):
#         state_est,_ = kf.filter(val_outputs[i, :, :].squeeze().detach().cpu().numpy())
#         val_samples.append(state_est)
#     val_samples = np.stack(val_samples)
#
#     # Make explicit index list for private dims
#     num_stateDims = A.shape[1]
#     private_dims = []
#     public_dims = []
#     dim_labels = []
#     for i in range(num_stateDims):
#         if isdim_priv[i]:
#             private_dims.append(i)
#             dim_labels.append(f"Private Dim {i}")
#         else:
#             public_dims.append(i)
#             dim_labels.append(f"Public Dim {i}")
#
#
#
#     ### Train the VAE
#     criterion = nn.MSELoss()  # TODO: Change this to something else
#     loss_list = []
#     eval_data = []
#     batch_count = int(tstates.shape[0] / batch_size)
#     train_layout = TrainLayout(epochs, batch_count, loss_list, eval_data)
#     with Live(train_layout.layout, console=console, refresh_per_second=1) as live:
#         for e in range(epochs):
#             epoch_loss = 0
#             for b in range(batch_count):
#                 ## BatchdWise Data
#                 t_cur_state = trn_states[b * batch_size : (b + 1) * batch_size].to(device)
#                 cur_output = trn_outputs[b * batch_size : (b + 1) * batch_size].to(
#                     device
#                 )
#                 if t_cur_state.shape[0] != batch_size:
#                     continue; # We are nat dealing with smaller sizes.
#                 ## Change Data to HideStuff
#                 masked_output = vae(cur_output)
#                 state_estimates_w_vae = []
#                 state_estimates_wo_vae = []
#
#                 logger.debug(f"Tell em about the shape of the output {cur_output.shape} as well as its type {type(cur_output)}")
#                 logger.debug(f"Shape of masked_output is {masked_output.shape} as well as its type {type(masked_output)}")
#
#                 # Go Through Batch
#                 for i in range(cur_output.shape[0]):
#                     logger.debug(f"Going through the batch {i}")
#                     # First without VAE
#                     (filtered_mean, filtered_covariance) = kf.filter(
#                         cur_output[i, :, :].squeeze().detach().cpu().numpy()
#                     )
#                     (smoothed_mean, smoothed_covariance) = kf.smooth(
#                         cur_output[i, :, :].squeeze().detach().cpu().numpy()
#                     )
#                     state_estimates_wo_vae.append(smoothed_mean)
#
#                 state_estimates_wo_vae = torch.from_numpy(np.array(state_estimates_wo_vae)).to(device)
#                 logger.debug(f"Check if tensor requires grad {cur_output.requires_grad}")
#                 cur_output.requires_grad = True
#                 state_estimates_w_vae = torch_filter(masked_output)
#                 logger.debug("Done with the batch. Will add more stuff in a minute")
#
#                 # CHECK: This will likely need a torch based implementation to 
#                 # have the computation graph involved
#                 logger.debug("Before loss estimation")
#                 similarities = F.mse_loss(state_estimates_w_vae[:,:,public_dims], t_cur_state[:,:,public_dims])
#                 diff = - F.mse_loss(state_estimates_wo_vae[:,:,private_dims], t_cur_state[:,:,private_dims])
#                 # similarities = F.mse_loss(state_estimates_w_vae[:,:,1:], t_cur_state[:,:,1:])
#                 # diff = - F.mse_loss(state_estimates_wo_vae[:,:,0], t_cur_state[:,:,0])
#                 final_loss = similarities + diff
#                 fl_mean = final_loss.mean()
#                 loss_list.append(fl_mean.item())
#                 cur_loss = final_loss.mean().item()
#                 
#                 logger.debug("Before optimizing.")
#                 optimizer.zero_grad()
#                 fl_mean.backward()
#                 optimizer.step()
#
#
#                 # Plot and Save the Reconstruction of Validation Samples
#                 inputo = vae(t_val_outputs[:batch_size, :, :])
#                 val_state_vae_est = torch_filter(inputo).squeeze().cpu().detach().numpy()
#                 func_to_compare = np.stack(
#                     [
#                         t_val_states[:num_valsamps, :, :].squeeze().cpu().detach().numpy(),
#                         val_state_vae_est[:num_valsamps, :, :],
#                         val_samples[:num_valsamps, :, :],
#                     ]
#                 ).transpose(1, 0, 2, 3)
#                 plot_functions(
#                     func_to_compare,
#                     f"./figures/vae_fixed_recon/plot_vaerecon_eval_{e:02d}_{b:02d}_recon_states.png",
#                     function_labels=["True Value","ITL Method","Control Method"],
#                     dim_labels=dim_labels[:2],
#                     dims_to_show=[0,1]
#                 )
#
#                 ## Now compare the outputs by saving their plot
#                 func_to_compare = np.stack(
#                     [
#                         cur_output[:2].squeeze().cpu().detach().numpy(),
#                         masked_output[:2].squeeze().cpu().detach().numpy(),
#                     ]
#                 ).transpose(1, 0, 2, 3)
#                 plot_functions_2by1(
#                     func_to_compare,
#                     f"./figures/vae_fixed_recon/plot_vaerecon_eval_{e:02d}_{b:02d}_recon_outputs.png",
#                     function_labels=["Ground Truth","ITL Method"],
#                     dim_to_show=[1,],
#                 )
#
#
#                 ## TODO: We need some sort of knob here to play with utility vs privacy
#                 train_layout.update(e, b, cur_loss, None)
#
#             # Normal Reporting
#             if (e + 1) % 1 == 0:
#                 print("Epoch: {}, Loss: {:.5f}".format(e + 1, epoch_loss))
#
#     # We test with MSE for reconstruction for now



def train_w_metrics(
    model: nn.Module,
    dataset: Tuple[np.ndarray, np.ndarray],
    epochs: int,
    batch_size: int,
    saveplot_dest: str,
    train_percent: float,
    device: torch.device,
):
    tlosses = []
    vlosses = []
    train_size = int(len(dataset[0]) * train_percent)
    batch_count = int(train_size // batch_size)
    tdataset = (dataset[0][:train_size, :], dataset[1][:train_size, :])
    vdataset = (dataset[0][train_size:, :], dataset[1][train_size:, :])

    # layout, progress = make_layout(0, tlosses, vlosses)
    layout = TrainLayout(epochs, batch_count, tlosses, vlosses)
    batch_num = 0
    with Live(layout.layout, console=console, refresh_per_second=10) as live:
        for epoch, batch_no, tloss, vloss in train(
            model, tdataset, vdataset, epochs, batch_size, saveplot_dest, device
        ):
            batch_num += 1
            logger.debug(f"Batch number: {batch_num}")
            tlosses.append(tloss)
            vlosses.append(vloss)
            layout.update(epoch, batch_no, tloss, vloss)

    # TODO: Recover this
    # Plot the losses after the episode finishes
    # t_diff_in_order = np.max(tlosses) - np.min(tlosses) > 1e1
    # v_diff_in_order = np.max(vlosses) - np.min(vlosses) > 1e1
    # fig, axs = plt.subplots(1, 2)
    # axs[0].plot(tlosses)
    # # if t_diff_in_order:
    # # axs[0].set_yscale("log")
    # axs[0].set_title("Training Loss")
    # axs[1].plot(vlosses)
    # # if v_diff_in_order:
    # #     axs[1].set_yscale("log")
    # axs[1].set_title("Validation Loss")
    # plt.show()
    #

# TODO: Later change the name of the function
def train_new(
    model: nn.Module,
    dataset: Tuple[List[str], List[np.ndarray]],
    epochs: int,
    batch_size: int,
    saveplot_dest: str,
    train_percent: float,
    device: torch.device,
):
    # Dataset comes preloaded as a DF


    raise NotImplementedError
    # We have to train this new batch of things
    # Dataset is divided on -> Runs, Coluns(time-wise features)

    # Test Set Should be the division of each of the runs into differnt amounts. But then this would mean that we are enforcing a particular batch of time to be hidden.
    # Which could contain stuff that is not native to our own case 

# TODO: We need to implement federated learning in this particular part of the expression
def federated():
    # We also need a federated aspect to all this. And its getting close to being time to implementing this
    raise NotImplementedError

class RecoveryNet(nn.Module):

    def __init__(self, input_size, state_size, num_outputs, time_steps):
        super().__init__()
        self.mean = torch.zeros(input_size, time_steps)
        self.variance = torch.zeros(input_size, time_steps)
        self.rnn = torch.nn.GRU(input_size, state_size, batch_first=True)
        # Final output layer
        self.output_layer = torch.nn.Linear(state_size, num_outputs)
        self.count = 0
        self.batch_norm = torch.nn.BatchNorm1d(num_features=input_size)

    def forward(self, x):
        # Normalize x
        # self.update(x)
        # normed_x = self.batch_norm(x)
        # norm_x = (x - self.mean) / (self.variance + 1e-8).sqrt()
        transposed_x = x.transpose(1, 2)
        logger.debug(f"Tranposed x looks like {transposed_x}")
        rnnout, hidden = self.rnn(transposed_x)
        logger.debug(f"RNN output looks like: {rnnout}")
        return self.output_layer(F.relu(rnnout)), hidden

    def update(self, x):
        self.count += 1
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        if self.count == 1:
            self.mean = batch_mean
        else:
            old_mean = self.mean
            self.mean = (old_mean * (self.count - 1) + batch_mean) / self.count
            delta = batch_mean - old_mean
            self.variance = (self.variance * (self.count - 1) + batch_var) / self.count

            # self.variance = (
            #     self.variance * (self.count - 1)
            #     + (x - old_mean - delta).pow(2).sum(dim=0)
            # ) / self.count



if __name__ == "__main__":

    args = argsies()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(int(time.time()))
    logger = create_logger("main_training")

    # TODO: Make it  so that generate_dataset checks if params are the same
    colums, runs_dict = load_defacto_data(args.defacto_data_raw_path)

    # Separate them into their splits
    train_runs, val_runs, test_runs = split_defacto_runs(
        runs_dict,
        train_split=0.8,
        val_split=0.2,
        test_split=0.0,
    )



    logger.debug(f"Columns are {colums}")
    logger.debug(f"Runs dict is {runs_dict}")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # type: ignore

    # With the Dataset in Place we Also Generate a Variational Autoencoder
    # vae = train_VAE(outputs) # CHECK: Should we train this first or remove for later
    vae = train_new():

    # 🚩 Development so far🚩
    exit()

    # TODO: We might want to do a test run here 
    # if len(test_runs) > 0:
    #     test_runs = train_new(test_runs)

    # trainVAE_wprivacy(
    #     training_data,
    #     (hidden, outputs),
    #     args.epochs,
    #     args.saveplot_dest,
    #     [True,False,True]
    # )
    #
if __name__ == "__main__":
    logger = create_logger("main_vae")
    main()
