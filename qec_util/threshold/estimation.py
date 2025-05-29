from collections.abc import Iterable, Callable

import pathlib
import numpy as np

from .util import get_fit_func, rescale_input, save_fit_information


def get_threshold(
    data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    fit_func_name: str = "poly-2",
    num_samples_bootstrap: int = 1000,
    conf_level: float = 0.975,
    params_guess: np.ndarray | None = None,
    weighted: bool = True,
    file_name: str | pathlib.Path | None = None,
) -> tuple[float, float, float]:
    """Returns the threshold estimation and its confidence intervals using
    bootstrapping.

    Parameters
    ----------
    data
        Dictionary with keys corresponding to the circuit distance, and
        values corresponding to the tuple of (1) physical error probabilities,
        (2) number of failures, (3) number of samples.
    fit_func_name
        Name of the function to be used for fitting. The options are ``"poly-i"``
        for a polynomial of order ``"i"``, and ``"tanh"``. See Notes for more
        information about these methods. By default ``"poly-2"``.
    num_samples_bootstrap
        Number of bootstrapping samples to take to estimate the confidence intervals.
        See Notes for more information about the bootstrapping procedure.
        If ``num_samples_bootstrap = 0``, the confidence intervals are not estimated.
        By default ``1000``.
    conf_level
        Confidence level used when bootstrapping. If ``num_samples_bootstrap = 0``,
        then the confidence bounds returned correspond to one standard deviation.
        By default ``0.975``.
    params_guess
        Guess for the free parameters in the fit function specified by ``fit_func_name``
        and ``p_threshold`` and ``mu`` used in the rescaling. The vector is passed as
        ``(p_threshold, mu, *params_for_fit_Func)``.
    weighted
        Use the standard deviation of the logical error probabilities when performing
        the fit.
    file_name
        Stores all information from the fit in a '.txt' file. If ``None``, it does
        not store all information. This information is usef for e.g. plotting
        the fit results, see ``qec_util.threshold.plot_threshold_fit``.
        By default ``None``.

    Returns
    -------
    p_threshold
        Estimated threshold (corresponding to a physical error probability).
    ci_lower
        Lower value of the confidence interval. See ``conf_level``.
    ci_upper
        Upper value of the confidence interval. See ``conf_level``.

    Notes
    -----
    The algorithm for estimating the threshold is from

        C. Wang, J. Harrington, and J. Preskill, Confinement-higgs transition in
        a disordered gauge theory and the accuracy threshold for quantum memory,
        Annals of Physics 303, 31 (2003)

    and

        J. W. Harrington, Analysis of Quantum Error-Correcting
        Codes: Symplectic Lattice Codes and Toric Codes, Ph.D.
        thesis, California Institute of Technology (2004).

    These two references use the ``"poly-i"`` fit function, while the ``"tanh"``
    has been described in

        Hillmann, T., Dauphinais, G., Tzitrin, I., & Vasmer, M.
        Single-shot and measurement-based quantum error correction via fault complexes.
        arXiv preprint arXiv:2410.12963 (2024).

    The use of ``"tanh"`` is preferred if the logical error probabilities around
    the threshold have values close to its saturation (i.e. ``1 - (0.5)**k`` with
    ``k`` the number of logical qubits).

    The algorithm for fitting the function is ``scipy.optimize.curve_fit``,
    which uses least squares.

    The bootstrap estimation of the confidence interval has been reproduced from

    https://gist.github.com/chubbc/59f7dcb8d5d6f7a1a8f5ccc237c8d61d

    It assumes that the decoding failures follow a Bernouilli distribution
    (same as for the estimation of the logical error probability with the Wilson
    interval) and that the estimated logical error probability follows a beta
    distribution (see https://en.wikipedia.org/wiki/Conjugate_prior and
    https://en.wikipedia.org/wiki/Beta_distribution for more information).
    Instead of resampling the data (as done in the standard bootstrapping method),
    this function samples new estimates of the logical error probability
    using the beta distribution with ``alpha = num_failures`` and ``beta = num_success``.
    This is useful for when the logical error probability is very close to 0.

    The choice of the beta distribution is based on the fact that its expected
    value corresponds to ``alpha/(alpha + beta)``, which mimics the expected
    value of the Bernouilli distribution, and its variance is
    ``alpha/(alpha + beta) * beta/(alpha + beta) * 1/(alpha + beta)``,
    which mimics the variance when performing ``alpha + beta`` Bernouilli trials.

    Notes
    -----
    This function requires ``scipy``. To install the requirements to be able
    to execute any function in ``qec_util``, run ``pip install qec_util[all]``.
    """
    if not isinstance(data, dict):
        raise TypeError(f"'data' must be a dict, but {type(data)} was given.")
    if any(not isinstance(d, int) for d in data):
        raise TypeError("Each key of 'data' must be an int.")
    if any((not isinstance(v, Iterable)) or len(v) != 3 for v in data.values()):
        raise TypeError("Each value of 'data' must be a tuple of length 3.")
    for p, f, n in data.values():
        p, f, n = np.array(p), np.array(f), np.array(n)
        if not (p.shape == f.shape == n.shape):
            raise ValueError(
                "Each numpy array triplet in the values of 'data' must have same shape."
            )
        if not (len(p.shape) == len(f.shape) == len(n.shape) == 1):
            raise ValueError(
                "Each numpy array triplet in the values of 'data' must be a vector."
            )
    fit_func, num_params = get_fit_func(fit_func_name)
    if (not isinstance(num_samples_bootstrap, int)) or num_samples_bootstrap < 0:
        raise TypeError(
            "'num_samples_bootstrap' must be a non-negative int, "
            f"but {num_samples_bootstrap} was given."
        )
    if params_guess is None:
        params_guess = np.ones(num_params + 2)
        params_guess[0] = 0.01  # thresholds are usually around this value
    if (not isinstance(params_guess, Iterable)) or len(params_guess) != num_params + 2:
        raise TypeError(
            "'params_guess' does not have the same length as expected for "
            f"{fit_func_name} (i.e. {num_params + 2})."
        )

    import scipy.stats as stats

    # convert data into single numpy array so that the all the data is fitted
    # in the same fit. They can not be stored in a matrix because different
    # distances can have different number of physical error probabilities
    distances, phys_err, num_failures, num_samples = [], [], [], []
    for d, (ps, num_fails, num_succs) in data.items():
        distances += [d for _ in ps]
        phys_err += list(ps)
        num_failures += list(num_fails)
        num_samples += list(num_succs)
    distances = np.array(distances, dtype=np.float64)
    phys_err = np.array(phys_err, dtype=np.float64)
    num_failures = np.array(num_failures, dtype=np.float64)
    num_samples = np.array(num_samples, dtype=np.float64)

    beta_dists = [
        stats.beta(nf + 1, ns - nf + 1) for nf, ns in zip(num_failures, num_samples)
    ]
    log_err_std = np.array([dist.std() for dist in beta_dists])

    # run once with the given data to get a good params_guess and if
    # (1) num_samples_bootstrap = 0, or (2) file_name is not None
    popt, pcov = _least_square_fit(
        fit_func,
        phys_err,
        distances,
        num_failures / num_samples,
        p0=params_guess,
        sigma=log_err_std if weighted else None,
    )
    if num_samples_bootstrap == 0:
        if file_name is not None:
            save_fit_information(
                file_name=file_name, popt=popt, pcov=pcov, fit_func_name=fit_func_name
            )
        p_threshold, var = popt[0], pcov[0, 0]
        ci_lower = np.max([p_threshold - np.sqrt(var), 0])
        ci_upper = p_threshold + np.sqrt(var)
        return p_threshold, ci_lower, ci_upper

    p_thresholds = np.zeros(num_samples_bootstrap)
    params_guess = popt
    for k, _ in enumerate(p_thresholds):
        popt_bootstrap, _ = _least_square_fit(
            fit_func,
            phys_err,
            distances,
            log_err=np.array([dist.rvs() for dist in beta_dists]),
            p0=params_guess,
            sigma=log_err_std if weighted else None,
        )
        p_thresholds[k] = popt_bootstrap[0]

    p_threshold = p_thresholds.mean()
    p_thresholds.sort()
    ci_lower = p_thresholds[int(num_samples_bootstrap * (1 - conf_level))]
    ci_upper = p_thresholds[int(num_samples_bootstrap * conf_level)]
    if num_samples_bootstrap == 1:
        ci_lower, ci_upper = -np.inf, np.inf

    if file_name is not None:
        save_fit_information(
            file_name=file_name,
            popt=popt,
            pcov=pcov,
            fit_func_name=fit_func_name,
            bootstrap_thresholds=p_thresholds,
        )

    return p_threshold, ci_lower, ci_upper


def _least_square_fit(
    func: Callable,
    phys_err: np.ndarray,
    distances: np.ndarray,
    log_err: np.ndarray,
    sigma: np.ndarray | None = None,
    p0: np.ndarray | None = None,
    maxfev: int = 1_000_000,
    absolute_sigma: bool = True,
    **kargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the estimated threshold from the fit to the given
    function once rescaled.

    The optional parameters follow the same nomenclature as the optional
    parameters for ``scipy.optimize.curve_fit``.

    Notes
    -----
    This function requires ``scipy``. To install the requirements to be able
    to execute any function in ``qec_util``, run ``pip install qec_util[all]``.
    """
    from scipy.optimize import curve_fit

    def rescaled_func(x, *args):
        ps, ds = x[0], x[1]
        p_thr, mu, func_args = args[0], args[1], args[2:]
        return func(rescale_input(ps, ds, p_thr, mu), *func_args)

    popt, pcov = curve_fit(
        rescaled_func,
        (phys_err, distances),
        log_err,
        p0=p0,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        maxfev=maxfev,
        **kargs,
    )
    return popt, pcov
