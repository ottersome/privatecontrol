import torch
from torch import Tensor, nn


# Something has to take care of the states
# We might require this to be a module in on by itself
class Filter(nn.Module):
    def __init__(
        self,
        H_mat: Tensor,
        A_mat: Tensor,
        B_mat: Tensor,
        init_state: Tensor,  # CHECK: if we actually need the initial_state
        batch_size: int,
    ):
        self.A_Mat = A_mat
        self.B_mat = B_mat
        self.init_state = init_state

        self.state_size = H_mat.shape[0]
        self.obs_size = H_mat.shape[1]
        # INitialize Parameters needed for Estimation

    def _initialize_params(self,sequence_length):
        # Initialize K randomly
        # Initialie belief of x_o randomly
        # CHECK: If this has to be random initiation
        # Initiialize the First States
        
        # Initialize states
        states = torch.zeros(self.batch_size, sequence_length, self.state_size)
        Ps = torch.zeros(self.batch_size, sequence_length, self.state_size, self.state_size)

        # We initialize the first one gaussian at random
        states[:, 0, :] = torch.randn(self.batch_size, self.state_size)
        # CHECK: Is identitiy matrix the correct choice here?
        Ps[:, 0, :, :] = torch.eye(self.state_size)
        # P is the initial a posteriori covariance
        

        return states, Ps
         

    def forward(self, obs):
        """
        Will run a sequence of observations through the filter and 
        return the state estimate

        parameters
        ----------
            - obs:  Sting of observations one got from a simulation
                    Shape: (batch_size x sequence_length x obs_dimension)
        returns
        -------
            - TODO: Not sure yet here.
        """
        assert (
            obs.shape[0] == self.batch_size
        ), "Observation should be of shape (batch_size x sequence_length x obs_dimension)"
        assert (
            len(obs.shape) == 3
        ), "Observation should be of shape (batch_size x sequence_length x obs_dimension)"

        # We always start from scratch

        batch_size = obs.shape[0]
        sequence_length = obs.shape[1]
        obs_dimension = obs.shape[2]

        (
            states,
            Ps 
        ) = self._initialize_params(sequence_length)

        # We Start the initial states

        # Two states to this update.
        # First we calculate the state prediction
        for i in range(sequence_length-1):
            # What do we estimate the first state to be ?
            pass


    def _state_prediction(self) -> (Tensor, Tensor):
        """
        Will calculte projection for next state as well as the covariance
        """
        previous_state = self.current_state





def filter_update(
    state_prediction,
    state_covariance,
    observation=None,
    transition_matrix=None,
    transition_offset=None,
    transition_covariance=None,
    observation_matrix=None,
    observation_offset=None,
    observation_covariance=None,
):
    pass
