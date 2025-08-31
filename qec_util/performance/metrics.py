import lmfit

import numpy as np
from uncertainties import ufloat, Variable


def logical_error_prob(predictions: np.ndarray, true_values: np.ndarray) -> np.float64:
    """Returns the logical error probability.

    Parameters
    ----------
    predictions
        Predictions made by the decoder.
    true_values
        True values.

    Returns
    -------
    float
        Logical error probability, defined as
        ``# correct predictions / # total samples``.
    """
    return np.mean(predictions ^ true_values)


def logical_error_prob_decay(
    rounds: np.ndarray, error_rate: float, round_offset: int | float = 0
) -> np.ndarray:
    """Returns the theoretical logical error probability given the QEC round
    and the logical error rate per QEC cycle.

    Parameters
    ----------
    rounds
        Number of QEC rounds executed so far.
    error_rate
        Logical error rate per QEC cycle.
    round_offset
        Offset for the ``rounds``.

    Returns
    -------
    float | np.ndarray
        Theoretical logical error probability.

    Notes
    -----
    The reference for this expression is:

        O’Brien, T. E., Tarasinski, B., & DiCarlo, L. (2017).
        npj Quantum Information, 3(1), 39.
        https://arxiv.org/abs/1703.04136
    """
    return 0.5 * (1 - (1 - 2 * error_rate) ** (rounds - round_offset))


class LogicalErrorProbDecayModel(lmfit.model.Model):
    """
    Model to fit the logical error probability decay as a function
    of the number of QEC rounds.
    """

    def __init__(self, vary_round_offset: bool = True):
        super().__init__(logical_error_prob_decay)

        # configure constraints that are independent from the data to be fitted
        self.set_param_hint("error_rate", min=0, max=0.5, vary=True)
        self.set_param_hint("round_offset", value=0, vary=vary_round_offset)

        return

    def guess(
        self, data: np.ndarray, x: np.ndarray, **kws
    ) -> lmfit.parameter.Parameters:
        # to ensure they are np.ndarrays
        x, data = np.array(x), np.array(data)

        # guess parameters based on the data
        deriv_data = (data[1:] - data[:-1]) / (x[1:] - x[:-1])
        data_averaged = 0.5 * (data[1:] + data[:-1])
        error_rate_guess = 0.5 * (1 - np.exp(np.average(deriv_data / data_averaged)))

        self.set_param_hint("error_rate", value=error_rate_guess)

        params = self.make_params()

        return lmfit.models.update_param_vals(params, self.prefix, **kws)

    def fit(
        self,
        data: np.ndarray,
        rounds: np.ndarray,
        min_round_fit: int | float = 0,
        *args,
        **kargs,
    ) -> lmfit.model.ModelResult:
        """
        Fits the data to the model.

        Parameters
        ----------
        data
            Logical error probabilities in array-like format.
        rounds
            Number of QEC rounds in array_like format.
        min_round_fit
            Minimum QEC round to perform the fit to.

        Returns
        -------
        lmfit.model.ModelResult
            Result of the fit.
        """
        # to ensure they are np.ndarrays
        rounds, data = np.array(rounds), np.array(data)

        data = data[np.where(rounds >= min_round_fit)]
        rounds = rounds[np.where(rounds >= min_round_fit)]
        return super().fit(data, rounds=rounds, *args, **kargs)


def lmfit_par_to_ufloat(param: lmfit.parameter.Parameter) -> Variable:
    """
    Safe conversion of an :class:`lmfit.parameter.Parameter` to
    :code:`uncertainties.ufloat(value, std_dev)`.

    Parameters
    ----------
    param
        Parameter from ``lmfit``.

    Returns
    -------
    uncertainties.Variable
        Same parameter as a ``ufloat`` object.
    """
    value = param.value
    stderr = np.nan if param.stderr is None else param.stderr
    return ufloat(value, stderr)


def get_error_rate(
    rounds: np.ndarray,
    log_error_probs: np.ndarray,
    min_round_fit: int | float = 0,
    return_round_offset: bool = False,
) -> Variable | tuple[Variable, Variable]:
    """
    Fit the logical error probability to an exponential decay and returns
    the logical error rate per QEC round (and the round offset if specified).

    Parameters
    ----------
    rounds
        Number of rounds for each ``log_error_probs`` element.
    log_error_probs
        Logical error probabilities when running a given experiment a different
        amount of rounds.
    min_round_fit
        Minimum QEC round to start performing the fit. The specified number is
        included in the fit.
    return_round_offset
        Returns the ``round_offset`` parameter in ``logical_error_prob_decay``
        together with the logical error rate per QEC round.
    """
    rounds = np.array(rounds)
    log_error_probs = np.array(log_error_probs)

    # fit only from distance
    guess = LogicalErrorProbDecayModel().guess(log_error_probs, rounds)
    fit = LogicalErrorProbDecayModel().fit(
        log_error_probs, rounds, guess=guess, min_round_fit=min_round_fit
    )

    error_rate = lmfit_par_to_ufloat(fit.params["error_rate"])
    round_offset = lmfit_par_to_ufloat(fit.params["round_offset"])

    if return_round_offset:
        return error_rate, round_offset

    return error_rate


def confidence_interval_binomial(
    num_failures: int | np.ndarray,
    num_samples: int | np.ndarray,
    probit: float = 1.96,
    method="wilson",
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Returns the lower and upper bounds for the logical error probability
    given the number of decoding failures and samples.

    The lower and upper bounds are absolute (not relative to the average),
    meaning that :math:`lower_bound < num_failures/num_samples < upper_bound`.

    Parameters
    ----------
    num_failures
        Number of decoding failures.
    num_samples
        Number of samples.
    probit
        :math:`1 - \\alpha/2` quantile of a standard normal distribution
        corresponding to the target error rate :math:`\\alpha`.
        By default, assumes a 95% confidence interval (:math:`alpha = 0.05`).
    method
        Method to use to compute the confidence interval.
        The options are: ``"wilson"``.

    Returns
    -------
    lower_bound
        Lower bound of the confidence interval.
    upper bound
        Upper bound of the confidence interval.

    Notes
    -----
    The expressions have been extracted from the "Binomial proportion confidence
    interval" article from Wikipedia:
    ``https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#``
    """
    if method != "wilson":
        raise ValueError(
            f"Only the 'wilson' method is available, but '{method}' was given."
        )

    num_successes = num_samples - num_failures
    middle_point = (num_failures + 0.5 * probit**2) / (num_samples + probit**2)
    width = (
        probit
        / (num_samples + probit**2)
        * np.sqrt(num_successes * num_failures / num_samples + probit**2 / 4)
    )

    lower_bound = middle_point - width
    upper_bound = middle_point + width

    return lower_bound, upper_bound
