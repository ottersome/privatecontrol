"""
Disclaiimer: This code is taken from 

github.com:sktime/sktime/libs/pykalman/standard.py
"""
import numpy as np
from torch import nn

class KalmanFilter(nn.Module):
    r"""Implements the Kalman Filter, Kalman Smoother, and EM algorithm.

    This class implements the Kalman Filter, Kalman Smoother, and EM Algorithm
    for a Linear Gaussian model specified by,

    .. math::

        x_{t+1}   &= A_{t} x_{t} + b_{t} + \\text{Normal}(0, Q_{t}) \\\\
        z_{t}     &= C_{t} x_{t} + d_{t} + \\text{Normal}(0, R_{t})

    The Kalman Filter is an algorithm designed to estimate
    :math:`P(x_t | z_{0:t})`.  As all state transitions and observations are
    linear with Gaussian distributed noise, these distributions can be
    represented exactly as Gaussian distributions with mean
    `filtered_state_means[t]` and covariances `filtered_state_covariances[t]`.

    Similarly, the Kalman Smoother is an algorithm designed to estimate
    :math:`P(x_t | z_{0:T-1})`.

    The EM algorithm aims to find for
    :math:`\\theta = (A, b, C, d, Q, R, \\mu_0, \\Sigma_0)`

    .. math::

        \\max_{\\theta} P(z_{0:T-1}; \\theta)

    If we define :math:`L(x_{0:T-1},\\theta) = \\log P(z_{0:T-1}, x_{0:T-1};
    \\theta)`, then the EM algorithm works by iteratively finding,

    .. math::

        P(x_{0:T-1} | z_{0:T-1}, \\theta_i)

    then by maximizing,

    .. math::

        \\theta_{i+1} = \\arg\\max_{\\theta}
            \\mathbb{E}_{x_{0:T-1}} [
                L(x_{0:T-1}, \\theta)| z_{0:T-1}, \\theta_i
            ]

    Parameters
    ----------
    transition_matrices : [n_timesteps-1, n_dim_state, n_dim_state] or \
    [n_dim_state,n_dim_state] array-like
        Also known as :math:`A`.  state transition matrix between times t and
        t+1 for t in [0...n_timesteps-2]
    observation_matrices : [n_timesteps, n_dim_obs, n_dim_state] or [n_dim_obs, \
    n_dim_state] array-like
        Also known as :math:`C`.  observation matrix for times
        [0...n_timesteps-1]
    transition_covariance : [n_dim_state, n_dim_state] array-like
        Also known as :math:`Q`.  state transition covariance matrix for times
        [0...n_timesteps-2]
    observation_covariance : [n_dim_obs, n_dim_obs] array-like
        Also known as :math:`R`.  observation covariance matrix for times
        [0...n_timesteps-1]
    transition_offsets : [n_timesteps-1, n_dim_state] or [n_dim_state] \
    array-like
        Also known as :math:`b`.  state offsets for times [0...n_timesteps-2]
    observation_offsets : [n_timesteps, n_dim_obs] or [n_dim_obs] array-like
        Also known as :math:`d`.  observation offset for times
        [0...n_timesteps-1]
    initial_state_mean : [n_dim_state] array-like
        Also known as :math:`\\mu_0`. mean of initial state distribution
    initial_state_covariance : [n_dim_state, n_dim_state] array-like
        Also known as :math:`\\Sigma_0`.  covariance of initial state
        distribution
    random_state : optional, numpy random state
        random number generator used in sampling
    em_vars : optional, subset of ['transition_matrices', \
    'observation_matrices', 'transition_offsets', 'observation_offsets', \
    'transition_covariance', 'observation_covariance', 'initial_state_mean', \
    'initial_state_covariance'] or 'all'
        if `em_vars` is an iterable of strings only variables in `em_vars`
        will be estimated using EM.  if `em_vars` == 'all', then all
        variables will be estimated.
    n_dim_state: optional, integer
        the dimensionality of the state space. Only meaningful when you do not
        specify initial values for `transition_matrices`, `transition_offsets`,
        `transition_covariance`, `initial_state_mean`, or
        `initial_state_covariance`.
    n_dim_obs: optional, integer
        the dimensionality of the observation space. Only meaningful when you
        do not specify initial values for `observation_matrices`,
        `observation_offsets`, or `observation_covariance`.
    """

    def __init__(
        self,
        transition_matrices=None,
        observation_matrices=None,
        transition_covariance=None,
        observation_covariance=None,
        transition_offsets=None,
        observation_offsets=None,
        initial_state_mean=None,
        initial_state_covariance=None,
        random_state=None,
        em_vars=None,
        n_dim_state=None,
        n_dim_obs=None,
    ):
        """Initialize Kalman Filter."""
        # determine size of state space
        n_dim_state = _determine_dimensionality(
            [
                (transition_matrices, array2d, -2),
                (transition_offsets, array1d, -1),
                (transition_covariance, array2d, -2),
                (initial_state_mean, array1d, -1),
                (initial_state_covariance, array2d, -2),
                (observation_matrices, array2d, -1),
            ],
            n_dim_state,
        )
        n_dim_obs = _determine_dimensionality(
            [
                (observation_matrices, array2d, -2),
                (observation_offsets, array1d, -1),
                (observation_covariance, array2d, -2),
            ],
            n_dim_obs,
        )

        self.transition_matrices = transition_matrices
        self.observation_matrices = observation_matrices
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.transition_offsets = transition_offsets
        self.observation_offsets = observation_offsets
        self.initial_state_mean = initial_state_mean
        self.initial_state_covariance = initial_state_covariance
        self.random_state = random_state
        self.em_vars = em_vars
        self.n_dim_state = n_dim_state
        self.n_dim_obs = n_dim_obs

        if em_vars is None:
            self._em_vars = [
                "transition_covariance",
                "observation_covariance",
                "initial_state_mean",
                "initial_state_covariance",
            ]
        else:
            self._em_vars = em_vars

    def sample(self, n_timesteps, initial_state=None, random_state=None):
        r"""Sample a state sequence :math:`n_{\\text{timesteps}}` timesteps in length.

        Parameters
        ----------
        n_timesteps : int
            number of timesteps

        Returns
        -------
        states : [n_timesteps, n_dim_state] array
            hidden states corresponding to times [0...n_timesteps-1]
        observations : [n_timesteps, n_dim_obs] array
            observations corresponding to times [0...n_timesteps-1]
        """
        (
            transition_matrices,
            transition_offsets,
            transition_covariance,
            observation_matrices,
            observation_offsets,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()

        n_dim_state = transition_matrices.shape[-2]
        n_dim_obs = observation_matrices.shape[-2]
        states = np.zeros((n_timesteps, n_dim_state))
        observations = np.zeros((n_timesteps, n_dim_obs))

        # logic for instantiating rng
        if random_state is None:
            rng = check_random_state(self.random_state)
        else:
            rng = check_random_state(random_state)

        # logic for selecting initial state
        if initial_state is None:
            initial_state = rng.multivariate_normal(
                initial_state_mean, initial_state_covariance
            )

        # logic for generating samples
        for t in range(n_timesteps):
            if t == 0:
                states[t] = initial_state
            else:
                transition_matrix = _last_dims(transition_matrices, t - 1)
                transition_offset = _last_dims(transition_offsets, t - 1, ndims=1)
                transition_covariance = _last_dims(transition_covariance, t - 1)
                cov = newbyteorder(transition_covariance, "=")
                states[t] = (
                    np.dot(transition_matrix, states[t - 1])
                    + transition_offset
                    + rng.multivariate_normal(np.zeros(n_dim_state), cov)
                )

            observation_matrix = _last_dims(observation_matrices, t)
            observation_offset = _last_dims(observation_offsets, t, ndims=1)
            observation_covariance = _last_dims(observation_covariance, t)
            observations[t] = (
                np.dot(observation_matrix, states[t])
                + observation_offset
                + rng.multivariate_normal(
                    np.zeros(n_dim_obs), newbyteorder(observation_covariance, "=")
                )
            )

        return (states, np.ma.array(observations))

    def filter(self, X):
        r"""Apply the Kalman Filter.

        Apply the Kalman Filter to estimate the hidden state at time :math:`t`
        for :math:`t = [0...n_{\\text{timesteps}}-1]` given observations up to
        and including time `t`.  Observations are assumed to correspond to
        times :math:`[0...n_{\\text{timesteps}}-1]`.  The output of this method
        corresponding to time :math:`n_{\\text{timesteps}}-1` can be used in
        :func:`KalmanFilter.filter_update` for online updating.

        Parameters
        ----------
        X : [n_timesteps, n_dim_obs] array-like
            observations corresponding to times [0...n_timesteps-1].  If `X` is
            a masked array and any of `X[t]` is masked, then `X[t]` will be
            treated as a missing observation.

        Returns
        -------
        filtered_state_means : [n_timesteps, n_dim_state]
            mean of hidden state distributions for times [0...n_timesteps-1]
            given observations up to and including the current time step
        filtered_state_covariances : [n_timesteps, n_dim_state, n_dim_state] \
        array
            covariance matrix of hidden state distributions for times
            [0...n_timesteps-1] given observations up to and including the
            current time step
        """
        Z = self._parse_observations(X)

        (
            transition_matrices,
            transition_offsets,
            transition_covariance,
            observation_matrices,
            observation_offsets,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()

        (_, _, _, filtered_state_means, filtered_state_covariances) = _filter(
            transition_matrices,
            observation_matrices,
            transition_covariance,
            observation_covariance,
            transition_offsets,
            observation_offsets,
            initial_state_mean,
            initial_state_covariance,
            Z,
        )
        return (filtered_state_means, filtered_state_covariances)

    def filter_update(
        self,
        filtered_state_mean,
        filtered_state_covariance,
        observation=None,
        transition_matrix=None,
        transition_offset=None,
        transition_covariance=None,
        observation_matrix=None,
        observation_offset=None,
        observation_covariance=None,
    ):
        r"""Update a Kalman Filter state estimate.

        Perform a one-step update to estimate the state at time :math:`t+1`
        give an observation at time :math:`t+1` and the previous estimate for
        time :math:`t` given observations from times :math:`[0...t]`.  This
        method is useful if one wants to track an object with streaming
        observations.

        Parameters
        ----------
        filtered_state_mean : [n_dim_state] array
            mean estimate for state at time t given observations from times
            [1...t]
        filtered_state_covariance : [n_dim_state, n_dim_state] array
            covariance of estimate for state at time t given observations from
            times [1...t]
        observation : [n_dim_obs] array or None
            observation from time t+1.  If `observation` is a masked array and
            any of `observation`'s components are masked or if `observation` is
            None, then `observation` will be treated as a missing observation.
        transition_matrix : optional, [n_dim_state, n_dim_state] array
            state transition matrix from time t to t+1.  If unspecified,
            `self.transition_matrices` will be used.
        transition_offset : optional, [n_dim_state] array
            state offset for transition from time t to t+1.  If unspecified,
            `self.transition_offset` will be used.
        transition_covariance : optional, [n_dim_state, n_dim_state] array
            state transition covariance from time t to t+1.  If unspecified,
            `self.transition_covariance` will be used.
        observation_matrix : optional, [n_dim_obs, n_dim_state] array
            observation matrix at time t+1.  If unspecified,
            `self.observation_matrices` will be used.
        observation_offset : optional, [n_dim_obs] array
            observation offset at time t+1.  If unspecified,
            `self.observation_offset` will be used.
        observation_covariance : optional, [n_dim_obs, n_dim_obs] array
            observation covariance at time t+1.  If unspecified,
            `self.observation_covariance` will be used.

        Returns
        -------
        next_filtered_state_mean : [n_dim_state] array
            mean estimate for state at time t+1 given observations from times
            [1...t+1]
        next_filtered_state_covariance : [n_dim_state, n_dim_state] array
            covariance of estimate for state at time t+1 given observations
            from times [1...t+1]
        """
        # initialize matrices
        (
            transition_matrices,
            transition_offsets,
            transition_cov,
            observation_matrices,
            observation_offsets,
            observation_cov,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()
        transition_offset = _arg_or_default(
            transition_offset, transition_offsets, 1, "transition_offset"
        )
        observation_offset = _arg_or_default(
            observation_offset, observation_offsets, 1, "observation_offset"
        )
        transition_matrix = _arg_or_default(
            transition_matrix, transition_matrices, 2, "transition_matrix"
        )
        observation_matrix = _arg_or_default(
            observation_matrix, observation_matrices, 2, "observation_matrix"
        )
        transition_covariance = _arg_or_default(
            transition_covariance, transition_cov, 2, "transition_covariance"
        )
        observation_covariance = _arg_or_default(
            observation_covariance, observation_cov, 2, "observation_covariance"
        )

        # Make a masked observation if necessary
        if observation is None:
            n_dim_obs = observation_covariance.shape[0]
            observation = np.ma.array(np.zeros(n_dim_obs))
            observation.mask = True
        else:
            observation = np.ma.asarray(observation)

        predicted_state_mean, predicted_state_covariance = _filter_predict(
            transition_matrix,
            transition_covariance,
            transition_offset,
            filtered_state_mean,
            filtered_state_covariance,
        )
        (_, next_filtered_state_mean, next_filtered_state_covariance) = _filter_correct(
            observation_matrix,
            observation_covariance,
            observation_offset,
            predicted_state_mean,
            predicted_state_covariance,
            observation,
        )

        return (next_filtered_state_mean, next_filtered_state_covariance)

    def smooth(self, X):
        r"""Apply the Kalman Smoother.

        Apply the Kalman Smoother to estimate the hidden state at time
        :math:`t` for :math:`t = [0...n_{\\text{timesteps}}-1]` given all
        observations.  See :func:`_smooth` for more complex output

        Parameters
        ----------
        X : [n_timesteps, n_dim_obs] array-like
            observations corresponding to times [0...n_timesteps-1].  If `X` is
            a masked array and any of `X[t]` is masked, then `X[t]` will be
            treated as a missing observation.

        Returns
        -------
        smoothed_state_means : [n_timesteps, n_dim_state]
            mean of hidden state distributions for times [0...n_timesteps-1]
            given all observations
        smoothed_state_covariances : [n_timesteps, n_dim_state]
            covariances of hidden state distributions for times
            [0...n_timesteps-1] given all observations
        """
        Z = self._parse_observations(X)

        (
            transition_matrices,
            transition_offsets,
            transition_covariance,
            observation_matrices,
            observation_offsets,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()

        (
            predicted_state_means,
            predicted_state_covariances,
            _,
            filtered_state_means,
            filtered_state_covariances,
        ) = _filter(
            transition_matrices,
            observation_matrices,
            transition_covariance,
            observation_covariance,
            transition_offsets,
            observation_offsets,
            initial_state_mean,
            initial_state_covariance,
            Z,
        )
        (smoothed_state_means, smoothed_state_covariances) = _smooth(
            transition_matrices,
            filtered_state_means,
            filtered_state_covariances,
            predicted_state_means,
            predicted_state_covariances,
        )[:2]
        return (smoothed_state_means, smoothed_state_covariances)

    def em(self, X, y=None, n_iter=10, em_vars=None):
        """Apply the EM algorithm.

        Apply the EM algorithm to estimate all parameters specified by
        `em_vars`.  Note that all variables estimated are assumed to be
        constant for all time.  See :func:`_em` for details.

        Parameters
        ----------
        X : [n_timesteps, n_dim_obs] array-like
            observations corresponding to times [0...n_timesteps-1].  If `X` is
            a masked array and any of `X[t]`'s components is masked, then
            `X[t]` will be treated as a missing observation.
        n_iter : int, optional
            number of EM iterations to perform
        em_vars : iterable of strings or 'all'
            variables to perform EM over.  Any variable not appearing here is
            left untouched.
        """
        Z = self._parse_observations(X)

        # initialize parameters
        (
            self.transition_matrices,
            self.transition_offsets,
            self.transition_covariance,
            self.observation_matrices,
            self.observation_offsets,
            self.observation_covariance,
            self.initial_state_mean,
            self.initial_state_covariance,
        ) = self._initialize_parameters()

        # Create dictionary of variables not to perform EM on
        if em_vars is None:
            em_vars = self._em_vars

        if em_vars == "all":
            given = {}
        else:
            given = {
                "transition_matrices": self.transition_matrices,
                "observation_matrices": self.observation_matrices,
                "transition_offsets": self.transition_offsets,
                "observation_offsets": self.observation_offsets,
                "transition_covariance": self.transition_covariance,
                "observation_covariance": self.observation_covariance,
                "initial_state_mean": self.initial_state_mean,
                "initial_state_covariance": self.initial_state_covariance,
            }
            em_vars = set(em_vars)
            for k in list(given.keys()):
                if k in em_vars:
                    given.pop(k)

        # If a parameter is time varying, print a warning
        for k, v in get_params(self).items():
            if k in DIM and (k not in given) and len(v.shape) != DIM[k]:
                warn_str = (
                    "{0} has {1} dimensions now; after fitting, "
                    + "it will have dimension {2}"
                ).format(k, len(v.shape), DIM[k])
                warnings.warn(warn_str, stacklevel=2)

        # Actual EM iterations
        for _ in range(n_iter):
            (
                predicted_state_means,
                predicted_state_covariances,
                kalman_gains,
                filtered_state_means,
                filtered_state_covariances,
            ) = _filter(
                self.transition_matrices,
                self.observation_matrices,
                self.transition_covariance,
                self.observation_covariance,
                self.transition_offsets,
                self.observation_offsets,
                self.initial_state_mean,
                self.initial_state_covariance,
                Z,
            )
            (
                smoothed_state_means,
                smoothed_state_covariances,
                kalman_smoothing_gains,
            ) = _smooth(
                self.transition_matrices,
                filtered_state_means,
                filtered_state_covariances,
                predicted_state_means,
                predicted_state_covariances,
            )
            sigma_pair_smooth = _smooth_pair(
                smoothed_state_covariances, kalman_smoothing_gains
            )
            (
                self.transition_matrices,
                self.observation_matrices,
                self.transition_offsets,
                self.observation_offsets,
                self.transition_covariance,
                self.observation_covariance,
                self.initial_state_mean,
                self.initial_state_covariance,
            ) = _em(
                Z,
                self.transition_offsets,
                self.observation_offsets,
                smoothed_state_means,
                smoothed_state_covariances,
                sigma_pair_smooth,
                given=given,
            )
        return self

    def loglikelihood(self, X):
        """Calculate the log likelihood of all observations.

        Parameters
        ----------
        X : [n_timesteps, n_dim_obs] array
            observations for time steps [0...n_timesteps-1]

        Returns
        -------
        likelihood : float
            likelihood of all observations
        """
        Z = self._parse_observations(X)

        # initialize parameters
        (
            transition_matrices,
            transition_offsets,
            transition_covariance,
            observation_matrices,
            observation_offsets,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
        ) = self._initialize_parameters()

        # apply the Kalman Filter
        (
            predicted_state_means,
            predicted_state_covariances,
            kalman_gains,
            filtered_state_means,
            filtered_state_covariances,
        ) = _filter(
            transition_matrices,
            observation_matrices,
            transition_covariance,
            observation_covariance,
            transition_offsets,
            observation_offsets,
            initial_state_mean,
            initial_state_covariance,
            Z,
        )

        # get likelihoods for each time step
        loglikelihoods = _loglikelihoods(
            observation_matrices,
            observation_offsets,
            observation_covariance,
            predicted_state_means,
            predicted_state_covariances,
            Z,
        )

        return np.sum(loglikelihoods)

    def _initialize_parameters(self):
        """Retrieve parameters if they exist, else replace with defaults."""
        n_dim_state, n_dim_obs = self.n_dim_state, self.n_dim_obs

        arguments = get_params(self)
        defaults = {
            "transition_matrices": np.eye(n_dim_state),
            "transition_offsets": np.zeros(n_dim_state),
            "transition_covariance": np.eye(n_dim_state),
            "observation_matrices": np.eye(n_dim_obs, n_dim_state),
            "observation_offsets": np.zeros(n_dim_obs),
            "observation_covariance": np.eye(n_dim_obs),
            "initial_state_mean": np.zeros(n_dim_state),
            "initial_state_covariance": np.eye(n_dim_state),
            "random_state": 0,
            "em_vars": [
                "transition_covariance",
                "observation_covariance",
                "initial_state_mean",
                "initial_state_covariance",
            ],
        }
        converters = {
            "transition_matrices": array2d,
            "transition_offsets": array1d,
            "transition_covariance": array2d,
            "observation_matrices": array2d,
            "observation_offsets": array1d,
            "observation_covariance": array2d,
            "initial_state_mean": array1d,
            "initial_state_covariance": array2d,
            "random_state": check_random_state,
            "n_dim_state": int,
            "n_dim_obs": int,
            "em_vars": lambda x: x,
        }

        parameters = preprocess_arguments([arguments, defaults], converters)

        return (
            parameters["transition_matrices"],
            parameters["transition_offsets"],
            parameters["transition_covariance"],
            parameters["observation_matrices"],
            parameters["observation_offsets"],
            parameters["observation_covariance"],
            parameters["initial_state_mean"],
            parameters["initial_state_covariance"],
        )

    def _parse_observations(self, obs):
        """Safely convert observations to their expected format."""
        obs = np.ma.atleast_2d(obs)
        if obs.shape[0] == 1 and obs.shape[1] > 1:
            obs = obs.T
        return obs


def _determine_dimensionality(variables, default):
    """Derive the dimensionality of the state space.

    Parameters
    ----------
    variables : list of ({None, array}, conversion function, index)
        variables, functions to convert them to arrays, and indices in those
        arrays to derive dimensionality from.
    default : {None, int}
        default dimensionality to return if variables is empty

    Returns
    -------
    dim : int
        dimensionality of state space as derived from variables or default.
    """
    # gather possible values based on the variables
    candidates = []
    for v, converter, idx in variables:
        if v is not None:
            v = converter(v)
            candidates.append(v.shape[idx])

    # also use the manually specified default
    if default is not None:
        candidates.append(default)

    # ensure consistency of all derived values
    if len(candidates) == 0:
        return 1
    else:
        if not np.all(np.array(candidates) == candidates[0]):
            raise ValueError(
                "The shape of all "
                + "parameters is not consistent.  "
                + "Please re-check their values."
            )
        return candidates[0]

def array2d(X, dtype=None, order=None):
    """Return at least 2-d array with data from X."""
    return np.asarray(np.atleast_2d(X), dtype=dtype, order=order)

def array1d(X, dtype=None, order=None):
    """Return at least 1-d array with data from X."""
    return np.asarray(np.atleast_1d(X), dtype=dtype, order=order)
