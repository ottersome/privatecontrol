from typing import Optional, Tuple
import torch
from torch import Tensor, nn


# Something has to take care of the states
# We might require this to be a module in on by itself
class Filter(nn.Module):
    def __init__(
        self,
        transition_matrix: Tensor,
        observation_matrix: Tensor,
        input_matrix: Tensor,
        process_noise_covariance: Tensor,
        measurement_noise_covariance: Tensor,
        initial_state_mean: Tensor,
        batch_size: int,
    ):
        self.A_Mat = transition_matrix
        self.B_mat = input_matrix
        self.C_mat = observation_matrix
        self.Q_mat = process_noise_covariance
        self.R_mat = measurement_noise_covariance
        self.input_size = input_matrix.shape[1]
        self.initial_state_mean = initial_state_mean

        self.state_size = observation_matrix.shape[0]
        self.obs_size = observation_matrix.shape[1]

    def _initialize_params(
        self, sequence_length: int, inputs: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:

        if inputs is None:
            inputs = torch.zeros(self.batch_size, sequence_length, self.input_size)

        # CHECK: If random initiation is correct
        states = torch.zeros(self.batch_size, sequence_length, self.state_size)
        Ps = torch.zeros(
            self.batch_size, sequence_length, self.state_size, self.state_size
        )

        # ~~We initialize the first one gaussian at random~~
        # states[:, 0, :] = torch.randn(self.batch_size, self.state_size)
        # Actually we want to use the initial state mean
        states[:, 0, :] = self.initial_state_mean
        # CHECK: Is identitiy matrix the correct choice here?
        # Prior is set to be of covariance of 1 meaning an assumption of independence
        Ps[:, 0, :, :] = torch.eye(self.state_size)

        return states, Ps, inputs
         

    def forward(self, obs, inputs: Optional[Tensor]):
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
        (states, P_post, inputs) = self._initialize_params(sequence_length, inputs)

        # Loop through current state id
        for cs_idx in range(1, sequence_length + 1):
            x_km1 = states[:, cs_idx - 1, :]  # Previous State
            u_k = inputs[:, cs_idx, :]  # Inputs
            z_k = obs[:, cs_idx, :]  # Observation

            # First Stage
            x_prior, P_prior = self._stage1_state_prediction(x_km1, u_k, P_post)
            # Second Stage
            x_post, P_post = self._stage2_measurement_update(P_prior, z_k, x_prior)

            # Update
            states[:, cs_idx, :] = x_post

        return states

    def _stage1_state_prediction(
        self, x_km1: Tensor, u_k: Tensor, P_post: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Will calculate projection for next state as well as the covariance
        """
        # CHECK: Ensure this is being done in a batch matmul manner
        x_prior = self.A_Mat @ x_km1 + self.B_mat @ u_k
        P_prior = self.A_mat @ P_post @ self.A_mat.T + self.Q_mat

        return x_prior, P_prior

    def _stage2_measurement_update(
        self, P_prior: Tensor, z_k: Tensor, xpri_k: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Will use observations to update information
        """
        # Calculate Kalman Gain
        K = (
            P_prior
            @ self.H_mat.T
            @ torch.inverse(self.H_mat @ P_prior @ self.H_mat.T + self.R_mat)
        )
        # Update the estimate with the measurement
        x_post = xpri_k + K @ (z_k - self.H_mat @ xpri_k)
        P_post = (torch.eye(self.state_size) - K @ self.H_mat) @ P_prior
        return x_post, P_post



