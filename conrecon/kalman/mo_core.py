import warnings
from typing import Optional, Tuple

from numpy.random import f
import torch
from torch import Tensor, nn

from conrecon.utils import create_logger

# Something has to take care of the states
# We might require this to be a module in on by itself
class Filter(nn.Module):
    def __init__(
        self,
        transition_matrix: Tensor,
        observation_matrix: Tensor,
        input_matrix: Tensor,
        initial_state_mean: Tensor,
        batch_size,
        process_noise_covariance: Optional[Tensor] = None,
        measurement_noise_covariance: Optional[Tensor] = None,
    ):
        super().__init__()
        self.A_mat = transition_matrix
        self.B_mat = input_matrix
        self.C_mat = observation_matrix
        self.Q_mat = process_noise_covariance
        self.R_mat = measurement_noise_covariance
        self.input_size = input_matrix.shape[1]
        self.initial_state_mean = initial_state_mean
        self.output_size = self.C_mat.shape[0]
        self.batch_size = batch_size
        self.state_size = transition_matrix.shape[0]
        self.obs_size = observation_matrix.shape[1]

        self.logger = create_logger(__class__.__name__)

        # Wwarn if Q and R are not provided
        if self.Q_mat == None:
            warnings.warn("Q is not provided. Will use identity matrix. This may lead to unexepected results")
            self.Q_mat = torch.eye(self.state_size)
        if self.R_mat == None:
            warnings.warn("R is not provided. Will use identity matrix. This may lead to unexepected results")
            self.R_mat = torch.eye(self.output_size)

    def _initialize_params(
        self, sequence_length: int, inputs: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:

        if inputs is None:
            inputs = torch.zeros(self.batch_size, sequence_length, self.input_size)

        # CHECK: If random initiation is correct
        states_est = torch.zeros(self.batch_size, sequence_length+1, self.state_size)
        Ps = torch.zeros(
            self.batch_size, sequence_length+1, self.state_size, self.state_size
        )

        # ~~We initialize the first one gaussian at random~~
        # states[:, 0, :] = torch.randn(self.batch_size, self.state_size)
        # Actually we want to use the initial state mean
        inp_tens = self.initial_state_mean.repeat(self.batch_size, 1,1)
        states_est[:, 0, :] =  self.initial_state_mean
        # CHECK: Is identitiy matrix the correct choice here?
        # Prior is set to be of covariance of 1 meaning an assumption of independence
        Ps[:, 0, :, :] = torch.eye(self.state_size)

        return states_est, Ps, inputs
         

    def forward(self, obs: Tensor, inputs: Optional[Tensor] = None) -> Tensor:
        """
        Will run a sequence of observations through the filter and
        return the state estimate

        Args
            obs:  Sting of observations one got from a simulation
                  Shape: (batch_size x sequence_length x obs_dimension)
        Returns:
            - TODO: Not sure yet here.
        """
        assert (
            obs.shape[0] == self.batch_size
        ), "Observation should be of shape (batch_size x sequence_length x obs_dimension)"
        assert (
            len(obs.shape) == 3
        ), "Observation should be of shape (batch_size x sequence_length x obs_dimension)"

        _ = obs.shape[0] # Batch Size
        sequence_length = obs.shape[1]


        # We Start the initial states
        (states_est, P_post, inputs) = self._initialize_params(sequence_length, inputs)

        # Loop through current state id
        for cs_idx in range(1, sequence_length + 1):
            x_km1 = states_est[:, cs_idx - 1, :]  # Previous Stat
            u_k = inputs[:, cs_idx-1, :]  # Inputs
            z_k = obs[:, cs_idx-1, :]  # Observation

            # First Stage
            x_prior, P_prior = self._stage1_state_prediction(x_km1, u_k, P_post[:,cs_idx-1,:,:])
            # Second Stage
            x_post, P_post_iter_est = self._stage2_measurement_update(P_prior, z_k, x_prior)
            P_post[:,cs_idx,:,:] = P_post_iter_est

            # Update
            states_est[:, cs_idx, :] = x_post

        return states_est

    def _stage1_state_prediction(
        self, x_km1: Tensor, u_k: Tensor, P_post: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Will calculate projection for next state as well as the covariance
        """
        assert isinstance(self.Q_mat, torch.Tensor), "Q should be a torch tensor"
        A_expanded = self.A_mat.unsqueeze(0)  # Shape becomes (1, 3, 3)
        B_expanded = self.B_mat.unsqueeze(0)  # Shape becomes (1, 3, 1)
        x_km1_unsq = x_km1.unsqueeze(2)
        uk_unsq = u_k.unsqueeze(2)
        # Now we can use torch.matmul to perform the batch multiplication
        # result = torch.matmul(batched_input, A_expanded)

        # x_prior = self.A_Mat @ x_km1 + self.B_mat @ u_k
        x_prior = (
            torch.matmul(A_expanded, x_km1_unsq)
            + torch.matmul(B_expanded, uk_unsq)
        ).squeeze(2)
        # Confirmed
        P_prior = self.A_mat @ P_post @ self.A_mat.T + self.Q_mat

        return x_prior, P_prior

    def _stage2_measurement_update(
        self, P_prior: Tensor, z_k: Tensor, xpri_k: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Will use observations to update information
        """
        # Calculate Kalman Gain
        # Unchecked:
        K = (
            P_prior
            @ self.C_mat.T
            @ torch.inverse(self.C_mat @ P_prior @ self.C_mat.T + self.R_mat)
        )
        z_k_unsq = z_k.unsqueeze(2)
        xpri_k_unsq = xpri_k.unsqueeze(2)
        C_expanded = self.C_mat.unsqueeze(0)
    
        # Update the estimate with the measurement
        # Numerically unchecked
        x_post = (xpri_k_unsq + K @ (z_k_unsq - C_expanded @ xpri_k_unsq)).squeeze(2)
        P_post = (torch.eye(self.state_size) - K @ C_expanded) @ P_prior
        return x_post, P_post



