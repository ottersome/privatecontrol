import numpy as np

def array2d(X, dtype=None, order=None):
    """Return at least 2-d array with data from X."""
    return np.asarray(np.atleast_2d(X), dtype=dtype, order=order)

def array1d(X, dtype=None, order=None):
    """Return at least 1-d array with data from X."""
    return np.asarray(np.atleast_1d(X), dtype=dtype, order=order)


def preprocess_arguments(argsets, converters):
    """Convert and collect arguments in order of priority.

    Parameters
    ----------
    argsets : [{argname: argval}]
        a list of argument sets, each with lower levels of priority
    converters : {argname: function}
        conversion functions for each argument

    Returns
    -------
    result : {argname: argval}
        processed arguments
    """
    result = {}
    for argset in argsets:
        for argname, argval in argset.items():
            # check that this argument is necessary
            if argname not in converters:
                raise ValueError(f"Unrecognized argument: {argname}")

            # potentially use this argument
            if argname not in result and argval is not None:
                # convert to right type
                argval = converters[argname](argval)

                # save
                result[argname] = argval

    # check that all arguments are covered
    if not len(converters.keys()) == len(result.keys()):
        missing = set(converters.keys()) - set(result.keys())
        s = f"The following arguments are missing: {list(missing)}"
        raise ValueError(s)

    return result

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "{0} cannot be used to seed a numpy.random.RandomState" + " instance"
    ).format(seed)

def newbyteorder(arr, new_order):
    """Change the byte order of an array.

    Parameters
    ----------
    arr : ndarray
        Input array.
    new_order : str
        Byte order to force.

    Returns
    -------
    arr : ndarray
        Array with new byte order.
    """
    if numpy2:
        return arr.view(arr.dtype.newbyteorder(new_order))
    else:
        return arr.newbyteorder(new_order)

def preprocess_arguments(argsets, converters):
    """Convert and collect arguments in order of priority.

    Parameters
    ----------
    argsets : [{argname: argval}]
        a list of argument sets, each with lower levels of priority
    converters : {argname: function}
        conversion functions for each argument

    Returns
    -------
    result : {argname: argval}
        processed arguments
    """
    result = {}
    for argset in argsets:
        for argname, argval in argset.items():
            # check that this argument is necessary
            if argname not in converters:
                raise ValueError(f"Unrecognized argument: {argname}")

            # potentially use this argument
            if argname not in result and argval is not None:
                # convert to right type
                argval = converters[argname](argval)

                # save
                result[argname] = argval

    # check that all arguments are covered
    if not len(converters.keys()) == len(result.keys()):
        missing = set(converters.keys()) - set(result.keys())
        s = f"The following arguments are missing: {list(missing)}"
        raise ValueError(s)

    return result

def log_multivariate_normal_density(X, means, covars, min_covar=1.0e-7):
    """Log probability for full covariance matrices."""
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probabily stuck in a component with too
            # few observations, we need to reinitialize this components
            cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim), lower=True)
        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = np.linalg.solve(cv_chol, (X - mu).T).T
        log_prob[:, c] = -0.5 * (
            np.sum(cv_sol**2, axis=1) + n_dim * np.log(2 * np.pi) + cv_log_det
        )

    return log_prob

def get_params(obj):
    """Get names and values of all parameters in `obj`'s __init__."""
    try:
        # get names of every variable in the argument
        args = inspect.getfullargspec(obj.__init__)[0]
        args.pop(0)  # remove "self"

        # get values for each of the above in the object
        argdict = {arg: obj.__getattribute__(arg) for arg in args}
        return argdict
    except Exception:
        raise ValueError("object has no __init__ method")
